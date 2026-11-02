# Proyecto Integrador: Segunda Entrega

## Implementación de un método Gauss-Newton libre de derivadas para mínimos cuadrados no lineales

**Artículo base:** *A derivative-free Gauss-Newton method* (Cartis y Roberts, 2019)
**Integrantes:** David Castro, Alejandro Pardo, Tomás Rincón
**Curso:** Métodos Numéricos · **Docente:** Edwin Fernando Muñoz
Universidad Externado de Colombia · Bogotá, 2026

---

## 1. Introducción y cambios respecto a la primera entrega

Implementamos una versión simplificada de DFO-GN para minimizar f(x) = ½‖r(x)‖² sin derivadas de r, y la comparamos con métodos que sí las usan o las aproximan. Atendemos las cuatro observaciones de la retroalimentación: **problemas de prueba** definidos (sección 3), **tolerancias** explícitas (sección 3), **generación del ruido** precisada (sección 3) y **elementos conservados** del algoritmo original (Tabla 1).

También hacemos un ajuste al plan: además del punto de Cauchy comprometido, agregamos un paso de Gauss-Newton con QR, porque los resultados mostraron que Cauchy no alcanza alta precisión. El método de Gauss-Newton se construye a partir de mínimos cuadrados con QR y la linealización del método de Newton.

## 2. Implementación

Todos los métodos de región de confianza comparten un solo núcleo, que combina el **origen de J** con el **tipo de paso**. Así, cada comparación aísla un solo efecto. El álgebra es propia; NumPy almacena los datos y SciPy solo valida.

- **Jacobiano por interpolación (ec. 2.4).** Con n+1 puntos se plantea W Jᵀ = D, donde W tiene filas yₜ − xₖ. Los m residuos comparten W, de modo que una eliminación de Gauss con pivoteo parcial resuelve todos los lados derechos. La razón entre el menor y el mayor pivote de W/Δ vigila la geometría: si baja de 10⁻⁸, los puntos se reconstruyen.
- **Paso de Cauchy.** Es s = −t g con t = min(gᵀg/gᵀHg, Δ/‖g‖), g = Jᵀr y H = JᵀJ: descenso del gradiente con la tasa óptima para el modelo.
- **Paso QR.** Resuelve min‖Js + r‖ con Gram-Schmidt modificado y Rs = −Qᵀr, evitando las ecuaciones normales, que elevan al cuadrado el condicionamiento. Se recorta a Δ. Una primera versión, solo recortada, se detuvo en f ≈ 32 en Freudenstein & Roth (mínimo: 24.49), porque con J casi singular puede incumplir el decrecimiento de Cauchy (ec. 2.8). Por eso se elige el paso, QR o Cauchy, que más baje el modelo.
- **Comparadores.** GN con J exacto; GN con diferencias finitas hacia adelante (h = √ε·max(1, |xⱼ|)); y descenso del gradiente con η fija, escogida como la mejor de {10⁻¹, …, 10⁻⁵} para cada problema sin ruido, lo que favorece a este método.

**Tabla 1. Elementos del Algoritmo 1 del artículo en nuestra versión.**

| Elemento | Estado | Justificación |
|---|---|---|
| Modelo lineal por interpolación (2.2–2.4) y modelo de f (2.5) | Conservado | Idea central del método |
| Región de confianza, cociente Rₖ con η₁ = 0.1 y η₂ = 0.7 | Conservado | Controla la confianza en el modelo |
| Actualización de Δₖ (2.15) | Simplificado (sin ρₖ) | Reduce parámetros |
| Parada (sección 4.3): f pequeña, Δ < 10⁻¹⁰, presupuesto | Conservado | Comparabilidad |
| Subproblema (rutina tipo DFBOLS/BOBYQA) | Sustituido por Cauchy o QR | Cumple la condición 2.8 del artículo |
| Geometría (Λ-poised) | Simplificado: razón de pivotes y reconstrucción | Criterio más simple de implementar |
| Fases de criticidad y de seguridad, radio ρₖ | Omitidos | Simplificación; efecto discutido en sección 5 |
| Actualización de factorización (sección 4.4) | Omitido | Con n ≤ 6 resolver desde cero es barato |

**Validación.** `gauss` y `QR` coinciden con NumPy en 200 sistemas aleatorios (error de QR y de ortogonalidad ≤ 8·10⁻¹⁶). Los siete problemas reproducen el 2f(x₀) de la Tabla 2 del artículo; J analítico coincide con diferencias centradas (error relativo ≤ 2.4·10⁻¹⁰); y `scipy.optimize.least_squares` reproduce los 2f*.

## 3. Diseño experimental

**Problemas** (Tabla 2 del artículo, de Moré, Garbow y Hillstrom, 1981; Pk es el problema k de esa tabla): P7 Rosenbrock (2,2), P13 Freudenstein & Roth (2,2; residuo no nulo), P9 Helical Valley (3,3), P15 Bard (3,15), P11 Powell Singular (4,4; J singular en la solución), P17 Kowalik & Osborne (4,11) y P19 Watson (6,31).

**Tolerancias.** Métrica del artículo (ec. 5.1): N(τ) es el número de evaluaciones de r hasta f(xₖ) ≤ f* + τ(f(x₀) − f*), con τ ∈ {10⁻¹, 10⁻³, 10⁻⁵}. Presupuesto: 200(n+1) evaluaciones.

**Ruido** (sección 5.2 del artículo). Multiplicativo r̃ᵢ = rᵢ(1+ε) y aditivo r̃ᵢ = rᵢ+ε, con ε ~ N(0, σ²), σ = 10⁻², independiente por componente y evaluación. Diez corridas con `np.random.default_rng(2026+k)`. El algoritmo solo ve el valor ruidoso; la métrica se mide con f real. No incluimos el ruido χ² del artículo. El cuaderno `03_experimentos.ipynb` regenera todas las tablas y figuras; dos ejecuciones dieron resultados idénticos, salvo los tiempos.

## 4. Resultados y comparación

**Tabla 2. N(τ = 10⁻⁵) sin ruido: evaluaciones de r (— = no alcanzado).** GN exacto requiere además J; DG usa J exacto.

| Problema | DFO-GN QR | DFO-GN Cauchy | GN exacto QR | GN exacto Cauchy | GN dif. finitas QR | DG |
|---|---:|---:|---:|---:|---:|---:|
| P7 Rosenbrock | 71 | — | 29 | — | 71 | — |
| P13 Freudenstein & Roth | 157 | — | 19 | — | 45 | — |
| P9 Helical Valley | 55 | 633 | 16 | 457 | 52 | — |
| P15 Bard | 14 | — | 6 | — | 24 | — |
| P11 Powell Singular | 26 | 371 | 7 | 214 | 35 | 729 |
| P17 Kowalik & Osborne | 35 | — | 8 | — | 36 | — |
| P19 Watson | 33 | — | 8 | — | 56 | — |

![Figura 1](../figuras/fig1_convergencia.png)

**Figura 1. Convergencia sin ruido:** mejor f(xₖ) − f* contra evaluaciones de r.

**El paso limita más que el origen de J.** La Tabla 2 y la Figura 1 muestran que Cauchy alcanza τ = 10⁻⁵ solo en 2 de 7 problemas, con J interpolado o exacto, y sus curvas son paralelas a las del descenso del gradiente: tener derivadas exactas no lo rescata. Es coherente, porque Cauchy es un paso de máximo descenso. Con QR, en cambio, DFO-GN resuelve los 7 problemas.

**DFO-GN frente a diferencias finitas.** Sin usar derivadas, DFO-GN QR necesitó menos evaluaciones en 4 de 7 problemas (Bard, Powell, Kowalik, Watson), empató en Rosenbrock y necesitó más en Helical Valley (por poco) y Freudenstein & Roth (3.5 veces). Reutilizar puntos cuesta 1.3–3.2 evaluaciones por iteración, frente a 1.9–5.0, pero exige más iteraciones.

**Orden de convergencia.** Con eₖ = ‖xₖ − x*‖, GN exacto muestra p ≈ 1.0 en Bard, Powell Singular, Kowalik y Watson, lo esperable con residuo no nulo o J singular. En Rosenbrock y Helical Valley hay muy pocos iterados en la zona asintótica para estimarlo, y en Freudenstein & Roth y en DFO-GN el estimador fue inestable, por lo que no sacamos conclusión.

**Tabla 3. Corridas con éxito, de 70 (7 problemas × 10 semillas), multiplicativo / aditivo.**

| Método | τ = 10⁻¹ | τ = 10⁻³ | τ = 10⁻⁵ |
|---|:--:|:--:|:--:|
| DFO-GN QR | 49 / 51 | 38 / 35 | 25 / 14 |
| DFO-GN Cauchy | 24 / 22 | 12 / 12 | 0 / 0 |
| GN dif. finitas QR | 0 / 0 | 0 / 0 | 0 / 0 |
| GN exacto QR (usa J exacto) | 70 / 62 | 64 / 60 | 57 / 44 |
| Descenso del gradiente (usa J exacto) | 70 / 70 | 40 / 40 | 10 / 10 |

![Figura 2](../figuras/fig2_ruido.png)

**Figura 2. Brecha final con ruido por problema, en 10 corridas.**

**Robustez.** Es el resultado más claro. GN con diferencias finitas fracasa en las 140 corridas y queda en f(x₀) (Figura 2): con h ≈ 10⁻⁸ y σ = 10⁻², el ruido en J es del orden de σ/h ≈ 10⁶, lo que confirma la motivación del artículo. DFO-GN QR, también sin derivadas, sí progresa, aunque con mucha dispersión: en Helical Valley y Powell Singular la Figura 2 muestra corridas casi perfectas junto a otras que no avanzan. Los métodos con J exacto son una referencia, no una alternativa real, porque en un problema sin derivadas no están disponibles.

**Tabla 4. Costo por iteración (rango entre problemas).**

| Método | Evaluaciones de r | Tiempo (ms) | Sistema resuelto |
|---|:--:|:--:|---|
| DFO-GN QR | 1.32–3.20 | 0.25–0.77 | Gauss n×n con m lados derechos + QR m×n |
| GN exacto QR | 1.00 (+ J) | 0.15–0.42 | QR m×n |
| GN dif. finitas QR | 1.87–5.00 | 0.16–0.52 | QR m×n |

La iteración de DFO-GN QR tarda más en tiempo de pared, porque hace Gauss y QR. Esto importa poco cuando r es costosa, que es el caso de interés: ahí domina el número de evaluaciones.

## 5. Análisis preliminar

**Fortalezas.** DFO-GN QR converge a los valores de referencia en los 7 problemas sin derivadas y, a diferencia de las diferencias finitas, tolera el ruido.

**Limitaciones.** (i) Cauchy solo no basta para alta precisión. (ii) Freudenstein & Roth consume todo el presupuesto, un problema con residuo no nulo. (iii) Con ruido, entre 60 y 70 de las 70 corridas de DFO-GN QR terminan por radio pequeño: sospechamos que los rechazos causados por el ruido encogen Δ, de modo que el error de J (≈ σ/Δ) crece. El radio ρₖ que omitimos existe justamente para acotar esa reducción; aún no lo hemos comprobado experimentalmente. (iv) Siete problemas no permiten generalizar.

**Tabla 5. Artículo frente a nuestra implementación.**

| Aspecto | Artículo | Nosotros | Consecuencia |
|---|---|---|---|
| Problemas | 53 | 7 | Sin perfiles de datos; comparación por problema |
| Comparadores | BOBYQA, DFBOLS, POUNDERS | GN exacto, GN-DF, DG | **No podemos contrastar** su ventaja de tiempo (7 a 444 veces frente a Py-DFBOLS) |
| Subproblema | Rutina DFBOLS/BOBYQA | Cauchy o QR + Cauchy | Cauchy solo es más lento |
| Geometría | Λ-poised | Razón de pivotes | Implementación simple; posiblemente más evaluaciones |
| Protección de Δ | ρₖ | Ninguna | Probable pérdida de robustez con ruido |

Nuestros resultados apoyan la motivación del artículo: la interpolación cuesta un número de evaluaciones comparable al de las diferencias finitas, sin fracasar con ruido. El artículo no hace esta comparación, sino que enfrenta a DFO-GN con otros métodos sin derivadas, de modo que no reproducimos sus cifras.

**Mejoras para la entrega final.** Incorporar ρₖ y medir su efecto con ruido (prioritaria); añadir los tiempos con r costosa simulada; probar más problemas y perfiles de datos; analizar el ruido con las herramientas de Monte Carlo (S25–S27).

## 6. Conclusiones preliminares

- Interpolar J con Gauss y dar el paso con QR produce un método sin derivadas que resuelve los 7 problemas a τ = 10⁻⁵ sin ruido.
- La elección del paso pesa más que el origen de J: Cauchy limita incluso con derivadas exactas.
- Con ruido, las diferencias finitas fracasan por completo y DFO-GN conserva capacidad de avance, aunque irregular.
- La mejora prioritaria es proteger el radio frente al ruido.

## Declaración de uso de inteligencia artificial

Usamos Claude (Anthropic), modelo Claude Opus 5.5, como apoyo en la redacción inicial del código en Python, la propuesta de los experimentos y el primer borrador de este informe. Los resultados provienen de la ejecución del código. Como grupo, analizamos línea por línea que todo siguiera los lineamientos del curso.

## Referencias

- Cartis, C., & Roberts, L. (2019). A derivative-free Gauss-Newton method. *Mathematical Programming Computation*, 11, 631–674.
- Moré, J. J., Garbow, B. S., & Hillstrom, K. E. (1981). Testing unconstrained optimization software. *ACM TOMS*, 7(1), 17–41.
- Moré, J. J., & Wild, S. M. (2009). Benchmarking derivative-free optimization algorithms. *SIAM Journal on Optimization*, 20(1), 172–191.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2.ª ed.). Springer.
- Turner, P. R., Arildsen, T., & Kavanagh, K. (2018). *Applied Scientific Computing With Python*. Springer.

## Anexo: código

Tres cuadernos ejecutables en Colab o Jupyter, en la carpeta `Segunda_Entrega/cuadernos` de https://github.com/AlejandroP212/metodos-numericos-dfo-gn : `01_metodos_y_validacion`, `02_dfogn_y_comparadores` y `03_experimentos`.
