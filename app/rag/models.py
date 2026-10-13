"""Contratos Pydantic de la capa RAG (entrada, fragmentos recuperados y respuesta validada)."""
from typing import List

from pydantic import BaseModel, Field, model_validator

NO_LO_SE = "No lo sé"


class ConsultaConocimiento(BaseModel):
    pregunta: str = Field(min_length=5, max_length=1000, description="Pregunta sobre la base de conocimiento interna.")


class FragmentoRecuperado(BaseModel):
    """Un fragmento (chunk) devuelto por el retriever, con su procedencia."""

    fuente: str = Field(description="Archivo de origen.")
    chunk: int = Field(description="Número de chunk dentro de la colección.")
    texto: str
    distancia: float = Field(description="Distancia coseno (menor = más parecido).")


class RespuestaRAG(BaseModel):
    """Respuesta basada exclusivamente en el contexto recuperado."""

    respuesta: str = Field(description=f"Respuesta en español basada SOLO en el contexto, o exactamente '{NO_LO_SE}'.")
    encontrado_en_contexto: bool = Field(description="true si la respuesta está en el contexto; false si no.")
    fuentes: List[str] = Field(default_factory=list, description="Archivos (source) de los fragmentos usados.")

    @model_validator(mode="after")
    def coherencia(self) -> "RespuestaRAG":
        # Si no está en el contexto, la respuesta es "No lo sé" y no se citan fuentes.
        if not self.encontrado_en_contexto:
            self.respuesta = NO_LO_SE
            self.fuentes = []
        return self
