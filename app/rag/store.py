"""Ingesta y recuperación sobre ChromaDB (persistente y local).

Diseño heredado de la Pre-entrega 3: IDs deterministas (hash) para no duplicar, un único modelo de
embeddings registrado como metadato de la colección y chunking por tokens con overlap.

Chroma es síncrono: desde el código `async` se lo llama con `asyncio.to_thread` / métodos `a*` de
LangChain para no bloquear el event loop de FastAPI.
"""
import asyncio
import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.rag.models import FragmentoRecuperado

logger = logging.getLogger("rag.store")


# ---------------------------------------------------------------- embeddings
def get_embeddings() -> Embeddings:
    """El MISMO modelo para indexar y para consultar (se configura por variable de entorno)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not config.GOOGLE_API_KEY:
        raise RuntimeError("Falta GOOGLE_API_KEY (ver .env.example).")
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=config.GOOGLE_API_KEY)


def embedding_model_id() -> str:
    return f"google/{config.EMBEDDING_MODEL}"


# ---------------------------------------------------------------- 1. carga y limpieza
def clean_text(text: str) -> str:
    """Normaliza espacios y saltos de línea sin romper los párrafos."""
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(knowledge_dir: Path | None = None) -> List[Document]:
    directorio = Path(knowledge_dir or config.KNOWLEDGE_DIR)
    docs = [
        Document(page_content=clean_text(p.read_text(encoding="utf-8")), metadata={"source": p.name})
        for p in sorted(directorio.glob("*"))
        if p.suffix.lower() in {".txt", ".md"}
    ]
    if not docs:
        raise FileNotFoundError(f"No hay archivos .txt o .md en '{directorio}'")
    logger.info("Documentos cargados: %d", len(docs))
    return docs


# ---------------------------------------------------------------- 2. chunking
def _splitter() -> RecursiveCharacterTextSplitter:
    separadores = ["\n\n", "\n", ". ", " ", ""]
    try:  # medir en tokens (tiktoken); requiere descargar el vocabulario la primera vez
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP, separators=separadores,
        )
    except Exception as exc:  # sin red / sin caché: aproximación por caracteres (~4 por token)
        logger.warning("tiktoken no disponible (%s): chunking por caracteres", type(exc).__name__)
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE * 4, chunk_overlap=config.CHUNK_OVERLAP * 4, separators=separadores,
        )


def split_documents(docs: List[Document]) -> List[Document]:
    chunks = _splitter().split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    logger.info("Chunks generados: %d (chunk_size=%d tokens, overlap=%d)", len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return chunks


def chunk_id(chunk: Document) -> str:
    """ID determinista: hash del origen + contenido. Permite reindexar sin duplicar."""
    raw = f"{chunk.metadata['source']}::{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------- 3. persistencia
def get_vectorstore() -> Chroma:
    """Abre (o crea) la colección persistente, registrando el modelo de embeddings usado."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "embedding_model": embedding_model_id()},
    )


def check_embedding_model(store: Chroma) -> None:
    """Evita consultar con un modelo de embeddings distinto al que se usó para indexar."""
    indexed_with = (store._collection.metadata or {}).get("embedding_model")
    if store._collection.count() and indexed_with and indexed_with != embedding_model_id():
        raise RuntimeError(
            f"La base fue indexada con '{indexed_with}' pero ahora se usa '{embedding_model_id()}'. "
            "Ejecutá: python -m app.rag.ingest --reset"
        )


def ingest(reset: bool = False) -> Chroma:
    """Indexa solo lo nuevo (idempotente). Con reset=True borra la colección y reindexa todo."""
    store = get_vectorstore()
    if reset and store._collection.count():
        logger.info("--reset: borrando la colección existente")
        store.delete_collection()
        store = get_vectorstore()
    check_embedding_model(store)

    chunks = split_documents(load_documents())
    ids = [chunk_id(c) for c in chunks]
    existentes = set(store.get(ids=ids, include=[])["ids"]) if store._collection.count() else set()
    nuevos = [(i, c) for i, c in zip(ids, chunks) if i not in existentes]
    if not nuevos:
        logger.info("La base ya está al día (%d chunks). No se reindexa nada.", store._collection.count())
        return store
    store.add_documents(documents=[c for _, c in nuevos], ids=[i for i, _ in nuevos])
    logger.info("Chunks nuevos indexados: %d | Total: %d", len(nuevos), store._collection.count())
    return store


async def aingest(reset: bool = False) -> Chroma:
    """Versión asíncrona: la ingesta (I/O + embeddings) corre en un hilo, sin bloquear el event loop."""
    return await asyncio.to_thread(ingest, reset)


# ---------------------------------------------------------------- 4. recuperación
@lru_cache(maxsize=1)
def _store_cacheado() -> Chroma:
    return get_vectorstore()


def reiniciar_cache() -> None:
    """Para tests: descarta el store cacheado (cambia la ruta o los embeddings)."""
    _store_cacheado.cache_clear()


def get_retriever(store: Chroma | None = None):
    return (store or _store_cacheado()).as_retriever(search_kwargs={"k": config.TOP_K})


async def buscar_fragmentos(consulta: str, k: int | None = None) -> List[FragmentoRecuperado]:
    """Recupera los k fragmentos más parecidos (async) con su distancia coseno."""
    store = _store_cacheado()
    pares = await store.asimilarity_search_with_score(consulta, k=k or config.TOP_K)
    return [
        FragmentoRecuperado(
            fuente=d.metadata.get("source", "?"), chunk=int(d.metadata.get("chunk", -1)),
            texto=d.page_content, distancia=float(score),
        )
        for d, score in pares
    ]
