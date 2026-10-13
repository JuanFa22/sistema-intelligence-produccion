# Protocolo de acciones críticas y aprobación humana (documento ficticio de demostración)

## Qué es una acción crítica
Es cualquier acción con efectos fuera del sistema: enviar correos o mensajes, publicar información, aplicar
cambios de precios o descuentos, ejecutar procesos o borrar datos. Las consultas de solo lectura y análisis no
son críticas.

## Regla de aprobación
Toda acción crítica requiere aprobación humana explícita antes de ejecutarse. El sistema debe pausar la ejecución,
mostrar la acción propuesta a un responsable y continuar solo si este aprueba. Si el responsable rechaza la acción,
el trabajo termina sin ejecutar nada y queda registrado el rechazo.

## Envío de reportes a gerencia
Los resúmenes para gerencia se envían únicamente a direcciones del dominio interno de la empresa. El envío queda
registrado en la bandeja de salida con destinatario y asunto, y el estado del trabajo conserva la aprobación
(quién la dio y cuándo) para auditoría.

## Trazabilidad
Cada ejecución debe poder reconstruirse completa: qué agente intervino, qué herramientas usó, qué fragmentos de
conocimiento recuperó y cuánto costó. Esta trazabilidad es obligatoria para los sistemas en producción.
