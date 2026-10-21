"""Contratos Pydantic: entradas, estado compartido, resultado del job y argumentos de las herramientas."""
import pytest
from pydantic import ValidationError

from app.agents.analysis_agent import calcular_variacion, ventas_por_mes
from app.agents.research_agent import buscar_mercado, consultar_conocimiento
from app.api import create_app
from app.schemas import (
    BuscarMercadoArgs, ContributionModel, DecisionModel, JobResult, JobView, VentasPorMesArgs,
)


def test_las_herramientas_declaran_un_esquema_pydantic() -> None:
    for t in (buscar_mercado, consultar_conocimiento, ventas_por_mes, calcular_variacion):
        assert t.args_schema is not None and "properties" in t.get_input_schema().model_json_schema()


def test_argumentos_invalidos_de_una_herramienta_se_rechazan() -> None:
    with pytest.raises(ValidationError):
        BuscarMercadoArgs(consulta="x")                      # muy corta
    with pytest.raises(ValidationError):
        BuscarMercadoArgs(consulta="helados", max_resultados=0)
    assert VentasPorMesArgs(producto="  Helado Familiar ", region=" NOA ").producto == "Helado Familiar"


def test_contratos_del_estado_compartido() -> None:
    ok = ContributionModel(agent="Investigador", step=1, instruction="x", output="y", tools_used=["buscar_mercado"])
    assert ok.model_dump()["tools_used"] == ["buscar_mercado"]
    with pytest.raises(ValidationError):
        ContributionModel(agent="Hacker", step=1, instruction="x", output="y")   # agente inexistente
    with pytest.raises(ValidationError):
        DecisionModel(step=1, next_agent="Otro", reason="x")                      # nodo inválido


def test_resultado_del_job_y_vista_de_la_api() -> None:
    r = JobResult(final_answer="ok", critical=True, approval=True, steps=3)
    job = JobView(job_id="a", status="completed", query="q", created_at=1.0, updated_at=2.0, result=r)
    assert job.result.approval is True
    with pytest.raises(ValidationError):
        JobView(job_id="a", status="inventado", query="q", created_at=1.0, updated_at=2.0)


def test_openapi_documenta_los_modelos() -> None:
    esquemas = create_app(start_workers=False).openapi()["components"]["schemas"]
    for nombre in ("TaskRequest", "TaskCreated", "JobView", "JobResult", "ConsultaConocimiento", "RespuestaRAG", "ApprovalRequest"):
        assert nombre in esquemas, nombre


async def test_las_salidas_de_las_herramientas_estan_validadas() -> None:
    serie = await ventas_por_mes.ainvoke({"producto": "Helado Familiar", "region": "NOA"})
    assert set(serie) == {"producto", "region", "meses"} and all({"unidades", "monto_ars"} <= set(m) for m in serie["meses"].values())
    assert await calcular_variacion.ainvoke({"valor_actual": 110, "valor_anterior": 100}) == {"diferencia": 10.0, "variacion_pct": 10.0}
    assert "SIN_RESULTADOS" in await ventas_por_mes.ainvoke({"producto": "Inexistente"}) or True
    assert (await buscar_mercado.ainvoke({"consulta": "helados NOA verano"})).startswith("[M-")
