# Capturas de evidencia (Arize Phoenix)

Guardá acá las capturas de tu corrida real (PNG), con estos nombres:

| Archivo | Qué debe verse |
|---|---|
| `01_traza_con_rag.png` | Traza completa del job donde el Investigador usa `consultar_conocimiento`: se ven el span de la herramienta y el span **retriever** (ChromaDB) con los fragmentos recuperados. |
| `02_traza_hitl.png` | Traza del job crítico: nodo `aprobacion_humana` (pausa, pestaña Events) y la reanudación `job-resume`. |
| `03_costo_por_ejecucion.png` | Lista de Traces con latencia, tokens y *Total Cost* de las 5 peticiones concurrentes. |
| `04_latencia_p95.png` | Gráfico *Trace latency* con el cuadro de percentiles (P50 y P95). |
| `05_consulta_rag_swagger.png` | (Opcional) `POST /knowledge/query` respondiendo en http://localhost:8000/docs, con su `fuentes`. |
