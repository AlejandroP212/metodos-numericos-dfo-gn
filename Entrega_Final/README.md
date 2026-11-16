# Entrega final: DFO-GN con protección del radio y análisis Monte Carlo

Proyecto Integrador de Métodos Numéricos (Externado, 2026-2). David Castro, Alejandro Pardo, Tomás Rincón.

## Contenido

| Ruta | Qué es |
|---|---|
| `informe/Informe_Final.pdf` | Informe final (fuente en `Informe_Final.md`) |
| `presentacion/presentacion.html` | Presentación interactiva de la sustentación. Se abre en el navegador, sin instalar nada |
| `cuadernos/` | Código: cuatro cuadernos autónomos |
| `resultados/` | Tablas que generan los cuadernos (CSV) |
| `figuras/` | Figuras del informe |
| `EXPLICACION_METODOS.md` | Cada método, de qué tema del curso sale y qué toma del artículo |

## Ejecutar

Descarga un cuaderno de `cuadernos/`, ábrelo en Colab o Jupyter y ejecuta todas las celdas. Cada uno funciona solo, sin otros archivos.

| Cuaderno | Contenido | Tiempo aprox. |
|---|---|---|
| `01_metodos_y_validacion.ipynb` | Gauss con pivoteo, QR, problemas de prueba, ruido y validación | segundos |
| `02_dfogn_final.ipynb` | DFO-GN por partes, con el paso QR + Cauchy y el radio inferior ρₖ | segundos |
| `03_experimentos_sin_ruido.ipynb` | N(τ), orden de convergencia, costo y Figura 1 | 1 min |
| `04_monte_carlo_ruido.ipynb` | 8 400 corridas con ruido, intervalos del 95 %, Figuras 2 y 3 | 6-10 min |

Semilla: `SEMILLA = 2026`. Versiones usadas: `requirements.txt`. Con otras versiones de las bibliotecas podrían cambiar los últimos decimales. Los tiempos cambian en cada ejecución.

## Qué respalda cada resultado

| Resultado del informe | Archivo |
|---|---|
| Tabla 3 y Figura 1 (sin ruido) | `resultados/e1_evaluaciones.csv`, `e1_curvas.csv` |
| Orden de convergencia | `resultados/e2_orden.csv` |
| Tabla 4 y Figura 2 (éxito con ruido) | `resultados/e3_exito_por_tau.csv`, `e3_resumen_exito.csv` |
| Tabla 5 (efecto de ρ) | `resultados/e3_diferencia_rho.csv` |
| Éxito por problema | `resultados/e3_exito_por_problema.csv` |
| Las 8 400 corridas | `resultados/e3_monte_carlo.csv` |
| Costo por iteración | `resultados/e4_costo.csv` |
| Tasa del descenso del gradiente | `resultados/e5_tasa_dg.csv` |
