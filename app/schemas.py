"""Contratos de datos (Pydantic v2) de TODO el sistema.

- Entradas y salidas de la API (`TaskCreated`, `JobView`, `HealthView`; las del RAG viven en `app/rag/models.py`).
- Contratos del estado compartido: cada nodo valida con `ContributionModel` / `DecisionModel` lo que escribe en
  `OrchestratorState` (el estado de LangGraph sigue siendo un TypedDict con reducers; la validación ocurre en la
  frontera de cada nodo y el resultado se guarda con `model_dump()`).
- Resultado final del job (`JobResult`): se valida antes de guardarse en Redis y de salir por la API.
- Argumentos de cada herramienta de los agentes (`args_schema`): el LLM no puede llamar a una tool con datos inválidos.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------- estado compartido (nodos del grafo)


class ContributionModel(BaseModel):
    """Aporte de un especialista (trazabilidad: quién, en qué paso, con qué herramientas)."""

    agent: Literal["Investigador", "Analista"]
    step: int = Field(ge=0)
    instruction: str
    output: str
    tools_used: list[str] = Field(default_factory=list)


class DecisionModel(BaseModel):
    """Decisión del supervisor, ya corregida por las guardas deterministas."""

    step: int = Field(ge=1)
    next_agent: Literal["Investigador", "Analista", "FINALIZAR"]
    reason: str
    instruction: str = ""
    guard: str | None = None


# ---------------------------------------------------------------- resultado final del job


class DecisionResumen(BaseModel):
    step: int
    next: str
    reason: str


class ContributionResumen(BaseModel):
    agent: str
    tools: list[str]


class JobResult(BaseModel):
    """Resultado que el worker guarda en Redis y la API devuelve."""

    final_answer: str = ""
    critical: bool = False
    approval: bool | None = None
    action_result: str = ""
    steps: int = Field(default=0, ge=0)
    decisions: list[DecisionResumen] = Field(default_factory=list)
    contributions: list[ContributionResumen] = Field(default_factory=list)


# ---------------------------------------------------------------- respuestas de la API


class TaskCreated(BaseModel):
    job_id: str
    status: str


class JobView(BaseModel):
    """Vista de un job. Admite campos extra (por ejemplo `interrupt` con la acción propuesta en el HITL)."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: Literal["pending", "running", "waiting_approval", "completed", "rejected", "FAILED"]
    query: str
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    duration_s: float | None = None
    result: JobResult | None = None
    error: str | None = None


class HealthView(BaseModel):
    status: str
    redis: bool
    observabilidad: str


# ---------------------------------------------------------------- argumentos de herramientas (args_schema)


class BuscarMercadoArgs(BaseModel):
    consulta: str = Field(min_length=2, description="Palabras clave de lo que se quiere investigar (ej. 'helados NOA verano').")
    max_resultados: int = Field(default=3, ge=1, le=10, description="Cantidad máxima de documentos (1 a 5; más de 5 se recorta).")


class ConsultarConocimientoArgs(BaseModel):
    consulta: str = Field(min_length=2, description="Pregunta o frase a buscar en la base de conocimiento interna, en lenguaje natural.")


class VentasPorMesArgs(BaseModel):
    producto: str = Field(min_length=2, description="Nombre EXACTO del producto ('Alfajor Triple', 'Galletitas Dulces' o 'Helado Familiar').")
    region: str | None = Field(default=None, description="Región opcional ('Centro' o 'NOA'). Si se omite suma todas.")

    @field_validator("producto", "region")
    @classmethod
    def _sin_espacios(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v


class CalcularVariacionArgs(BaseModel):
    valor_actual: float
    valor_anterior: float


# ---------------------------------------------------------------- salidas de las herramientas (validadas)


class DocumentoMercado(BaseModel):
    """Documento de la base de mercado (data/mercado.json)."""

    id: str = Field(pattern=r"^M-\d+$")
    titulo: str
    texto: str
    tags: list[str] = Field(default_factory=list)


class MesVentas(BaseModel):
    unidades: int = Field(ge=0)
    monto_ars: float = Field(ge=0)


class SerieVentas(BaseModel):
    """Salida de `ventas_por_mes`."""

    producto: str
    region: str
    meses: dict[str, MesVentas]


class Variacion(BaseModel):
    """Salida de `calcular_variacion`."""

    diferencia: float
    variacion_pct: float


def job_a_dict(model: BaseModel) -> dict[str, Any]:
    """Atajo: modelo -> dict listo para guardar en Redis / en el estado del grafo."""
    return model.model_dump()
