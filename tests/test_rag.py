"""Tests del RAG integrado (offline: embeddings y LLM falsos, sin API keys ni consumo de tokens)."""
import hashlib
import json
import re

import httpx
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from app import config
from app.api import create_app
from app.rag import store
from app.rag.chain import ServicioConocimiento, build_rag_chain
from app.rag.models import NO_LO_SE, ConsultaConocimiento, RespuestaRAG
from tests.test_api import patch_redis  # noqa: F401  (fixture)
from tests.fakes import fake_graph_factory


class BagOfWordsEmbeddings(Embeddings):
    """Embedding determinista (bolsa de palabras) para pruebas offline."""

    def _embed(self, text: str) -> list[float]:
        v = np.zeros(512)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def base(tmp_path, monkeypatch):
    """Base vectorial temporal con embeddings falsos y el conocimiento real de /knowledge."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "vectorstore")
    monkeypatch.setattr(store, "get_embeddings", lambda: BagOfWordsEmbeddings())
    store.reiniciar_cache()
    yield tmp_path
    store.reiniciar_cache()


def llm_json(**kw) -> FakeListChatModel:
    return FakeListChatModel(responses=[json.dumps(kw, ensure_ascii=False)])


# ------------------------------------------------------------------ ingesta
def test_limpieza_de_texto() -> None:
    assert store.clean_text("a\t\tb \r\n\r\n\r\n\r\nc  ") == "a b\n\nc"


def test_la_ingesta_indexa_y_es_idempotente(base) -> None:
    s = store.ingest()
    total = s._collection.count()
    assert total >= 5
    assert store.ingest()._collection.count() == total            # no duplica
    assert store.ingest(reset=True)._collection.count() == total   # reset reindexa todo


def test_ids_deterministas_y_metadatos(base) -> None:
    chunks = store.split_documents(store.load_documents())
    assert [store.chunk_id(c) for c in chunks] == [store.chunk_id(c) for c in store.split_documents(store.load_documents())]
    assert all({"source", "chunk"} <= set(c.metadata) for c in chunks)
    assert {c.metadata["source"] for c in chunks} >= {"01_politica_comercial_helados.md", "04_protocolo_acciones_criticas.md"}


def test_no_mezcla_modelos_de_embeddings(base, monkeypatch) -> None:
    store.ingest()
    monkeypatch.setattr(store, "embedding_model_id", lambda: "otro/modelo")
    with pytest.raises(RuntimeError, match="indexada con"):
        store.check_embedding_model(store.get_vectorstore())


# ------------------------------------------------------------------ recuperación (async)
async def test_recupera_el_fragmento_correcto_con_su_fuente(base) -> None:
    store.ingest()
    frs = await store.buscar_fragmentos("descuento por volumen de cajas por pedido")
    assert frs and frs[0].fuente == "01_politica_comercial_helados.md"
    assert len(frs) <= config.TOP_K and all(0 <= f.distancia <= 2 for f in frs)
    frs2 = await store.buscar_fragmentos("criterio de alineación con el mercado puntos porcentuales")
    assert "02_guia_estacionalidad_y_regiones.md" in {f.fuente for f in frs2}


async def test_la_herramienta_del_agente_cita_fuentes_KB(base) -> None:
    from app.agents.research_agent import consultar_conocimiento

    store.ingest()
    salida = await consultar_conocimiento.ainvoke({"consulta": "aprobación humana para enviar correos"})
    assert "[KB:04_protocolo_acciones_criticas.md#" in salida


async def test_herramienta_sin_base_devuelve_SIN_RESULTADOS(base) -> None:
    from app.agents.research_agent import consultar_conocimiento

    salida = await consultar_conocimiento.ainvoke({"consulta": "lo que sea"})   # base vacía
    assert salida.startswith("SIN_RESULTADOS")


# ------------------------------------------------------------------ cadena grounded + Pydantic
async def test_respuesta_con_fuentes_verificadas(base) -> None:
    s = store.ingest()
    chain = build_rag_chain(store.get_retriever(s), llm_json(
        respuesta="8% con aprobación de gerencia.", encontrado_en_contexto=True,
        fuentes=["01_politica_comercial_helados.md", "archivo_inventado.md"]))
    r = await ServicioConocimiento(chain).preguntar("¿Qué descuento hay desde 1000 cajas por pedido?")
    assert isinstance(r, RespuestaRAG) and r.encontrado_en_contexto
    assert "archivo_inventado.md" not in r.fuentes and "01_politica_comercial_helados.md" in r.fuentes


async def test_fuera_de_contexto_responde_no_lo_se(base) -> None:
    s = store.ingest()
    chain = build_rag_chain(store.get_retriever(s), llm_json(
        respuesta="Canberra", encontrado_en_contexto=False, fuentes=["01_politica_comercial_helados.md"]))
    r = await ServicioConocimiento(chain).preguntar("¿Cuál es la capital de Australia?")
    assert r.respuesta == NO_LO_SE and r.fuentes == []   # el validador de Pydantic fuerza la coherencia


def test_validacion_pydantic_de_la_entrada() -> None:
    with pytest.raises(ValidationError):
        ConsultaConocimiento(pregunta="no")


# ------------------------------------------------------------------ endpoint
class _ServicioFalso:
    async def preguntar(self, pregunta: str) -> RespuestaRAG:
        return RespuestaRAG(respuesta=f"eco: {pregunta}", encontrado_en_contexto=True, fuentes=["x.md"])


async def _post(app, path: str, body: dict) -> httpx.Response:
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            return await c.post(path, json=body)


async def test_endpoint_knowledge_query(patch_redis) -> None:  # noqa: F811
    app = create_app(graph_factory=fake_graph_factory(), knowledge=_ServicioFalso(), start_workers=False)
    r = await _post(app, "/knowledge/query", {"pregunta": "¿Cuál es la política de freezers?"})
    assert r.status_code == 200 and r.json()["fuentes"] == ["x.md"]


async def test_endpoint_valida_la_entrada_y_da_503_sin_base(patch_redis) -> None:  # noqa: F811
    app = create_app(graph_factory=fake_graph_factory(), knowledge=_ServicioFalso(), start_workers=False)
    assert (await _post(app, "/knowledge/query", {"pregunta": "no"})).status_code == 422
    sin_base = create_app(graph_factory=fake_graph_factory(), start_workers=False)
    assert (await _post(sin_base, "/knowledge/query", {"pregunta": "¿Cuál es la política de freezers?"})).status_code == 503


# ------------------------------------------------------------------ integración: agente -> RAG -> API
async def test_el_investigador_usa_el_rag_dentro_del_grafo(base, patch_redis) -> None:  # noqa: F811
    from tests.fakes import StatelessWorker
    from tests.test_api import _con_app, _esperar

    store.ingest()

    async def prueba(client, app):
        job_id = (await client.post("/tasks", json={"query": "Investigá helados NOA y analizá Helado Familiar"})).json()["job_id"]
        job = await _esperar(client, job_id, {"completed"})
        aporte = job["result"]["contributions"][0]
        assert aporte["agent"] == "Investigador" and aporte["tools"] == ["consultar_conocimiento"]

    await _con_app(fake_graph_factory(StatelessWorker(tool_name="consultar_conocimiento")), prueba)
