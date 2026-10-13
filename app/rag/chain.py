"""Cadena RAG asíncrona (LCEL): retriever -> prompt grounded -> LLM -> PydanticOutputParser -> verificación de fuentes."""
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.rag.models import NO_LO_SE, RespuestaRAG

parser = PydanticOutputParser(pydantic_object=RespuestaRAG)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un asistente de la empresa. Respondé ÚNICAMENTE con la información del CONTEXTO.\n"
        "Reglas:\n"
        "- No uses conocimiento propio ni inventes datos.\n"
        f"- Si la respuesta no está en el CONTEXTO, respondé exactamente \"{NO_LO_SE}\" y marcá "
        "encontrado_en_contexto como false.\n"
        "- En fuentes, listá el nombre de archivo de cada fragmento que usaste.\n\n"
        "{format_instructions}",
    ),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(format_instructions=parser.get_format_instructions())


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(sin fragmentos relevantes)"
    return "\n\n".join(f"[fuente: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)


def verificar_fuentes(data: dict) -> RespuestaRAG:
    """Las fuentes citadas deben ser archivos realmente recuperados (evita fuentes inventadas)."""
    respuesta: RespuestaRAG = data["respuesta"]
    recuperadas = list(dict.fromkeys(d.metadata.get("source", "?") for d in data["docs"]))
    if respuesta.encontrado_en_contexto:
        validas = [f for f in respuesta.fuentes if f in recuperadas]
        respuesta.fuentes = validas or recuperadas
    return respuesta


def build_rag_chain(retriever: Runnable, llm: BaseChatModel) -> Runnable:
    generacion = (
        {"contexto": itemgetter("docs") | RunnableLambda(format_docs), "pregunta": itemgetter("pregunta")}
        | prompt | llm | parser
    ).with_retry(retry_if_exception_type=(OutputParserException,), stop_after_attempt=2)  # JSON mal formado

    return (
        RunnablePassthrough.assign(docs=itemgetter("pregunta") | retriever)   # 1. recuperación (top_k)
        | RunnablePassthrough.assign(respuesta=generacion)                    # 2. generación grounded
        | RunnableLambda(verificar_fuentes)                                   # 3. salida validada
    )


class ServicioConocimiento:
    """Punto de entrada asíncrono del RAG: pregunta -> RespuestaRAG validada con fuentes."""

    def __init__(self, chain: Runnable) -> None:
        self._chain = chain

    async def preguntar(self, pregunta: str) -> RespuestaRAG:
        return await self._chain.ainvoke({"pregunta": pregunta})

    @classmethod
    async def crear(cls, llm: BaseChatModel | None = None) -> "ServicioConocimiento":
        """Ingesta (si falta algo) en un hilo y arma la cadena con el LLM compartido."""
        from app.graph import crear_modelo
        from app.rag.store import aingest, get_retriever

        store = await aingest()
        return cls(build_rag_chain(get_retriever(store), llm or crear_modelo()))
