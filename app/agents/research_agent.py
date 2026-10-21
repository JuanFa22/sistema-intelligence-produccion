"""Agente de Investigación: mercado (búsqueda local) + conocimiento interno vía RAG (ChromaDB)."""
import asyncio
import json
import re
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app import config
from app.agents.common import herramientas_usadas, respuesta_final
from app.rag.store import buscar_fragmentos
from app.schemas import BuscarMercadoArgs, ConsultarConocimientoArgs, ContributionModel, DocumentoMercado
from app.state import OrchestratorState

PROMPT = (
    "Sos el Agente de Investigación. Tu única tarea es buscar información con tus herramientas y "
    "resumirla con cifras y la fuente de cada dato:\n"
    "- buscar_mercado: tendencias, estacionalidad y benchmarks del MERCADO (citá el ID, por ejemplo [M-02]).\n"
    "- consultar_conocimiento: base de conocimiento INTERNA (RAG): políticas comerciales, criterios de "
    "alineación con el mercado, glosario, protocolos y fichas de producto (citá [KB:archivo#chunk]).\n"
    "Usá la herramienta que corresponda a lo que te piden (podés usar las dos). "
    "No calcules ventas internas ni inventes datos: si no encontrás algo, decilo. "
    "Respondé en español, en un máximo de 8 líneas."
)


@lru_cache(maxsize=1)
def _documentos() -> tuple[dict[str, Any], ...]:
    crudos = json.loads(config.MERCADO_JSON.read_text(encoding="utf-8"))
    return tuple(DocumentoMercado.model_validate(d).model_dump() for d in crudos)  # salida validada con Pydantic


def _tokens(texto: str) -> set[str]:
    return {t for t in re.findall(r"[a-záéíóúñ]+", texto.lower()) if len(t) > 2}


@tool(args_schema=BuscarMercadoArgs)
async def buscar_mercado(consulta: str, max_resultados: int = 3) -> str:
    """Busca documentos de investigación de mercado (consumo masivo, helados, golosinas, galletitas).

    Args:
        consulta: palabras clave de lo que se quiere investigar (ej. 'helados NOA verano').
        max_resultados: cantidad máxima de documentos a devolver (1 a 5).

    Devuelve los documentos más relevantes con su ID, título y texto. Si no hay
    coincidencias devuelve un mensaje que empieza con 'SIN_RESULTADOS'.
    """

    def _buscar() -> str:
        q = _tokens(consulta)
        puntuados = []
        for doc in _documentos():
            base = _tokens(doc["titulo"] + " " + doc["texto"]) | set(doc["tags"])
            puntaje = len(q & base) + 2 * len(q & set(doc["tags"]))
            if puntaje:
                puntuados.append((puntaje, doc))
        if not puntuados:
            return "SIN_RESULTADOS: probá con otras palabras clave (producto, región, categoría)."
        puntuados.sort(key=lambda x: -x[0])
        tope = max(1, min(max_resultados, 5))
        return "\n".join(f"[{d['id']}] {d['titulo']}: {d['texto']}" for _, d in puntuados[:tope])

    return await asyncio.to_thread(_buscar)


@tool(args_schema=ConsultarConocimientoArgs)
async def consultar_conocimiento(consulta: str) -> str:
    """Busca en la base de conocimiento INTERNA de la empresa (RAG sobre ChromaDB).

    Contiene: política comercial de helados (descuentos, pagos, freezers, devoluciones), guía de
    estacionalidad por región y criterio de alineación con el mercado, glosario de indicadores,
    protocolo de acciones críticas y ficha del producto Helado Familiar.

    Args:
        consulta: pregunta o frase a buscar, en lenguaje natural (ej. 'criterio para decir que un producto está alineado').

    Devuelve los fragmentos más relevantes con su fuente [KB:archivo#chunk]. Si no hay nada devuelve 'SIN_RESULTADOS'.
    """
    fragmentos = await buscar_fragmentos(consulta)
    if not fragmentos:
        return "SIN_RESULTADOS: la base de conocimiento no tiene información sobre eso."
    return "\n\n".join(f"[KB:{f.fuente}#{f.chunk}] {f.texto}" for f in fragmentos)


TOOLS = [buscar_mercado, consultar_conocimiento]


def crear_nodo_investigador(model: BaseChatModel):
    """Devuelve el nodo del grafo. El especialista recibe SOLO su instrucción (sin todo el historial)."""
    agente = create_react_agent(model, TOOLS, prompt=PROMPT, name="investigador")

    async def investigador_node(state: OrchestratorState) -> dict[str, Any]:
        paso = state.get("step_count", 0)
        instruccion = state["instruction"]
        resultado = await agente.ainvoke({"messages": [HumanMessage(content=instruccion)]})
        salida = respuesta_final(resultado["messages"]) or "Sin resultados."
        aporte = ContributionModel(  # contrato Pydantic del estado compartido
            agent="Investigador", step=paso, instruction=instruccion,
            output=salida, tools_used=herramientas_usadas(resultado["messages"]),
        ).model_dump()
        return {
            "messages": [HumanMessage(content=salida, name="Investigador")],
            "research_output": salida,
            "contributions": [aporte],
        }

    return investigador_node
