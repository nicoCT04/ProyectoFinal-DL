# Protocolo de prueba con usuarios (sección 4.6 del enunciado)

Objetivo: medir con personas reales los supuestos del modelo de valor que hoy son estimaciones
(`t_manual`, `p_manual`, `t_rev3`, `t_grabar`) y comparar **búsqueda manual vs. SeñaBusca**.
Duración: ~15 min por persona. Meta mínima: 5–8 participantes (compañeros de la U).

La app trae una **sesión guiada** que cronometra y registra todo: no hace falta cronómetro ni hoja
de registro.

## Preparación (una vez)

1. Videos de ejemplo de cada seña hechos por personas del conjunto de prueba (no se publican):
   `.venv/bin/python scripts/11_descargar_videos_prueba.py` (~350 MB).
2. Levantar la app: `.venv/bin/uvicorn app.server:app --port 8000`.
3. Abrir en Chrome **http://localhost:8000/?prueba=usuarios&debug=1** (`debug=1` guarda además los
   eventos de la segmentación y miniaturas en `logs/`, útil si algo falla).
4. Laptop sobre la mesa, luz de frente, la persona sentada con los hombros visibles.

## Sesión (la app guía cada paso)

1. Pedir consentimiento verbal y marcar la casilla. La app asigna el número (P1, P2…), el orden
   A→B / B→A (alterna entre participantes) y 3 + 3 señas de HELLO, THANKYOU, SCHOOL, UNDERSTAND,
   BATHROOM y TOMORROW, rotándolas para que cada seña pase por los dos métodos.
2. **Tarea A — búsqueda manual (3 señas).** La app muestra la seña hecha por otra persona (sin decir
   cuál es). «Empezar a buscar» inicia el reloj; la persona busca en otra pestaña (handspeak.com,
   lifeprint.com, signasl.org…), escribe lo que cree que significa y toca «La encontré» o «Me rindo»
   (máximo 5 min). La app muestra la respuesta correcta y el facilitador confirma si coincide.
3. **Tarea B — con SeñaBusca (3 señas).** La app muestra la seña; «Listo» inicia el reloj; la persona
   la imita frente a la cámara y toca «Es esta» en la opción correcta (o «Ninguna es»). Se registran
   los intentos, el modo (seguro / no seguro), la confianza, la opción elegida y su posición.
4. **Preguntas finales** (1–5): facilidad de cada método, confianza sin revisar el ejemplo, si la
   usaría en un curso y cuánto pagaría al mes, más un comentario libre.

## Validación propia

**http://localhost:8000/?prueba=propia&debug=1** recorre N señas del vocabulario (por defecto 20)
mostrando un video de ejemplo y el nombre: sirve para que el equipo pruebe la app con su propia
cámara antes de la presentación, con cada intento etiquetado.

## Cómo usar los resultados

* `.venv/bin/python scripts/12_resumen_prueba_usuarios.py` genera `reports/tablas/prueba_usuarios.json`
  y `docs/RESULTADOS_PRUEBA_USUARIOS.md` (anónimos) y muestra los supuestos medidos
  (`--excluir P3` descarta una sesión fallida).
* Sustituir esos valores en `asl/valor.py` → `Supuestos` y volver a correr `scripts/05_modelo_valor.py`
  y `scripts/08_numeros_informe.py`. El informe debe decir cuáles supuestos vienen de la prueba y
  cuáles siguen siendo estimaciones.
* `logs/` tiene datos de personas (registros y miniaturas): no se sube al repositorio (`.gitignore`).
