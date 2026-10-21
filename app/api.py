"""API REST (FastAPI) asíncrona: encola, no bloquea el event loop y devuelve un job_id. Incluye consulta RAG directa."""
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app import config, jobs
from app.jobs import JobStore
from app.observability import estado_observabilidad, iniciar_observabilidad
from app.rag.models import ConsultaConocimiento, RespuestaRAG
from app.redis_checkpointer import AsyncRedisCheckpointer
from app.schemas import HealthView, JobView, TaskCreated
from app.worker import WorkerPool


class TaskRequest(BaseModel):
    query: str = Field(min_length=5, max_length=2000, description="Consulta para el orquestador multi-agente.")


class ApprovalRequest(BaseModel):
    aprobado: bool
    comentario: str = ""


def create_app(
    graph_factory: Any = None,
    redis_url: str | None = None,
    start_workers: bool = True,
    knowledge: Any = None,
) -> FastAPI:
    """`graph_factory(checkpointer, executor) -> grafo` y `knowledge` (servicio RAG) permiten inyectar
    modelos falsos en los tests. Sin `graph_factory` se usan Gemini y la base ChromaDB reales."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        url = redis_url or config.REDIS_URL
        redis_txt = Redis.from_url(url, decode_responses=True)    # jobs, cola, outbox
        redis_bin = Redis.from_url(url, decode_responses=False)   # checkpoints (binario)
        checkpointer = AsyncRedisCheckpointer(redis_bin)

        async def ejecutor(accion: dict[str, Any]) -> str:
            # Efecto secundario REAL de la demo: registrar el email en la bandeja de salida (Redis).
            await redis_txt.rpush(config.OUTBOX_KEY, f"{accion['destinatario']}|{accion['asunto']}")
            return f"Email enviado a {accion['destinatario']} (registrado en outbox)"

        if graph_factory is not None:
            graph: CompiledStateGraph = graph_factory(checkpointer, ejecutor)
        else:
            from app.graph import construir_grafo, crear_modelo
            modelo = crear_modelo()
            graph = construir_grafo(modelo, checkpointer=checkpointer, action_executor=ejecutor)

        iniciar_observabilidad()
        app.state.knowledge = knowledge
        if knowledge is None and graph_factory is None:  # modo real: ingesta (si falta) + cadena RAG
            try:
                from app.rag.chain import ServicioConocimiento
                app.state.knowledge = await ServicioConocimiento.crear()
            except Exception as exc:  # la API arranca igual; /knowledge/query responde 503
                print(f"[API] RAG no disponible: {type(exc).__name__}: {exc}")
        store = JobStore(redis_txt)
        pool = WorkerPool(redis_txt, graph, store)
        app.state.store, app.state.redis, app.state.pool = store, redis_txt, pool
        if start_workers:
            pool.start()
        print(f"[API] {estado_observabilidad()} | workers: {pool.concurrency}")
        try:
            yield
        finally:
            await pool.stop()
            await redis_txt.aclose()
            await redis_bin.aclose()

    app = FastAPI(
        title="Sistema Intelligence de Producción",
        description="RAG (ChromaDB) + Supervisor multi-agente (LangGraph) + cola Redis + HITL + trazas Phoenix.",
        version="2.0.0", lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthView)
    async def health() -> HealthView:
        pong = await app.state.redis.ping()
        return HealthView(status="ok", redis=bool(pong), observabilidad=estado_observabilidad())

    @app.post("/knowledge/query")
    async def knowledge_query(body: ConsultaConocimiento) -> RespuestaRAG:
        """Consulta directa al RAG: recupera de ChromaDB y responde solo con el contexto (o 'No lo sé')."""
        if app.state.knowledge is None:
            raise HTTPException(status_code=503, detail="la base de conocimiento no está disponible")
        return await app.state.knowledge.preguntar(body.pregunta)

    @app.post("/tasks", status_code=202, response_model=TaskCreated)
    async def create_task(request: TaskRequest) -> TaskCreated:
        job_id = uuid.uuid4().hex[:12]
        await app.state.store.create(job_id, request.query)               # 1) estado inicial ANTES de encolar
        await app.state.store.enqueue({"job_id": job_id, "action": "start"})  # 2) cola Redis
        return TaskCreated(job_id=job_id, status=jobs.PENDING)             # 3) responde en milisegundos

    @app.get("/tasks/{job_id}", response_model=JobView, response_model_exclude_none=True)
    async def get_task(job_id: str) -> JobView:
        job = await app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job no encontrado")
        return JobView(**job)  # valida el estado que viene de Redis antes de responder

    @app.get("/status/{job_id}", response_model=JobView, response_model_exclude_none=True)  # alias del ejercicio de la Unidad 2
    async def get_status(job_id: str) -> JobView:
        return await get_task(job_id)

    @app.post("/tasks/{job_id}/approve", status_code=202, response_model=TaskCreated)
    async def approve_task(job_id: str, body: ApprovalRequest) -> TaskCreated:
        job = await app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job no encontrado")
        if job["status"] != jobs.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail=f"el job está en estado '{job['status']}', no espera aprobación")
        await app.state.store.update(job_id, status=jobs.RUNNING)
        await app.state.store.enqueue({"job_id": job_id, "action": "resume", "resume": body.model_dump()})
        return TaskCreated(job_id=job_id, status=jobs.RUNNING)

    return app


app = create_app()
