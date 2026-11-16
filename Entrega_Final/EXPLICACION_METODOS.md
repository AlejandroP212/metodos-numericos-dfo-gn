# Explicación de los métodos

Relación entre cada pieza del algoritmo, los temas de la asignatura y el artículo de Cartis & Roberts (2019).

## El problema en una línea

Minimizar f(x) = ½‖r(x)‖², con r: ℝⁿ → ℝᵐ y m ≥ n, **sin conocer las derivadas de r**.

## Mapa del algoritmo

| Paso de cada iteración | Tema relacionado | Sesión (microcurrículo) | Código |
|---|---|---|---|
| 1. Aproximar J con n+1 puntos | Interpolación + sistema lineal | S21, S11 | `JacobianoInterpolacion` |
| 2. Resolver W Jᵀ = D | Eliminación de Gauss, pivoteo, sustitución hacia atrás, idea de LU | S11, S12 | `eliminacion_gauss` |
| 3. Paso de Cauchy (artículo) | Se apoya en descenso del gradiente | S23 | `paso_cauchy` |
| 4. Paso de Gauss-Newton | Mínimos cuadrados + QR; Newton | S14, S18 | `minimos_cuadrados_qr` |
| 5. Región de confianza (artículo) | Optimización numérica | Artículo sección 2.2 | `region_confianza` |
| 6. Radio inferior ρₖ (artículo) | Control del tamaño de paso | Algoritmo 1, sección 4.5 | `region_confianza(..., proteger_radio=True)` |
| Ruido uniforme | Transformada inversa | S26 | `Evaluador` |
| Robustez con ruido | Monte Carlo, error estándar, intervalo del 95 % | S25–S27 | `corridas_monte_carlo`, `intervalo` |
| Comparador: diferencias finitas | Derivación numérica, épsilon, cancelación | S3, S4, S6 | `JacobianoDiferencias` |
| Comparador: descenso con η fija | Descenso del gradiente | S23 | `descenso_gradiente` |

## 1. Eliminación de Gauss (S11–S12)

**Base:** transformar Ax = b en Ux = c con m = aⱼᵢ/aᵢᵢ y aⱼₖ := aⱼₖ − m·aᵢₖ; luego, sustitución hacia atrás.

**En el proyecto:** por cada residuo rᵢ queremos un modelo lineal que pase por los puntos. Esto da n ecuaciones (ec. 2.4):

(y_t − x)ᵀ jᵢ = rᵢ(y_t) − rᵢ(x), para t = 1…n.

Los m sistemas tienen **la misma matriz W**: solo cambia el lado derecho. Por eso se elimina una sola vez y se aplica a las m columnas de D. Es el mismo argumento del Quiz 3 (ej. 2g) sobre la ventaja de LU.

**Pivoteo parcial.** Usamos pivoteo parcial porque W puede tener un cero o un número muy pequeño en la diagonal. Elegir el mayor |aⱼᵢ| asegura |m| ≤ 1, y así el redondeo no se amplifica.

**Costo:** ~n³/3 operaciones por la eliminación y ~n² por cada lado derecho.

**Si los puntos quedan casi alineados:** W casi singular → J poco confiable. Lo detectamos con la razón min|uᵢᵢ|/max|uᵢᵢ| < 10⁻⁸ y reconstruimos los puntos. Es condicionamiento (S5) aplicado.

## 2. Interpolación (S21)

**Base:** n+1 puntos determinan un polinomio único de grado ≤ n en una variable.

**En el proyecto:** n+1 puntos *bien repartidos* en ℝⁿ determinan un único modelo **lineal** en n variables: m(s) = r(x) + J s. Es la misma idea, con otra familia de funciones. La condición "xᵢ distintos" se convierte en "vectores y_t − x linealmente independientes".

**Por qué no se calculan derivadas:** J sale de exigir que el modelo pase por los puntos. Ningún paso usa la fórmula de r.

## 3. Mínimos cuadrados y QR (S14), y por qué se llama Gauss-Newton

**Base:** si Ax = b no tiene solución, se minimiza ‖Ax − b‖; con A = QR queda Rx = Qᵀb. Gram-Schmidt modificado actualiza vⱼ := vⱼ − rᵢⱼqᵢ.

**En el proyecto:** al linealizar, r(x+s) ≈ r + Js, y minimizar ½‖r + Js‖² **es un problema de mínimos cuadrados** con A = J y b = −r. Por eso el paso es: R s = −Qᵀr.

**¿Por qué QR y no ecuaciones normales?** JᵀJ s = −Jᵀr eleva al cuadrado el número de condición: cond(JᵀJ) = cond(J)².

**Relación con Newton (S18).** Newton para sistemas resuelve J Δx = −F. Si m = n y J es invertible, el paso de Gauss-Newton es *exactamente* ese. Si m > n, GN es la generalización: no se exige r = 0, solo que sea lo más pequeño posible.

## 4. Punto de Cauchy (artículo, sección 2.2, Supuesto 2.1)

El artículo exige que cada paso baje el modelo al menos tanto como una iteración de descenso con búsqueda exacta (ec. 2.8), y por eso lo usamos.

**Punto de partida (S23):** θ ← θ − η∇J, con η escogida a mano.

**Lo que cambia:** ∇f = g = Jᵀr, y η no se escoge a mano. Sobre la dirección −g, el modelo depende de un solo número t:

φ(t) = f − t‖g‖² + ½t²·gᵀHg, con H = JᵀJ.

Es una parábola. Su vértice es t\* = ‖g‖²/gᵀHg, pero el paso no puede salir de la región de radio Δ, así que t = min(t\*, Δ/‖g‖) y s = −t g.

**Ejemplo:** g = (2, 0), H = 2I. Entonces t\* = 4/8 = 0.5 y s = (−1, 0). Si Δ = 0.5, el tope es t = 0.25 y s = (−0.5, 0).

**Hallazgo:** es seguro pero lento, como el descenso del gradiente. Solo alcanza τ = 10⁻⁵ en 2 de 7 problemas, y lo mismo pasa con J exacto. El cuello de botella es el paso, no J.

## 5. Región de confianza (artículo, sección 2.2)

El modelo solo es bueno cerca de x. Se compara lo que prometió con lo que pasó:

R = (f(x) − f(x+s)) / (m(0) − m(s)).

| R | Significado | Acción |
|---|---|---|
| R ≥ 0.7 | El modelo acierta | Aceptar y agrandar Δ |
| 0.1 ≤ R < 0.7 | Acierta a medias | Aceptar y reducir Δ, sin bajar de ‖s‖ |
| R < 0.1 | El modelo falla | Rechazar y reducir Δ |

Cumple el papel de la tasa de aprendizaje, pero **se ajusta sola**.

**Lección de nuestra implementación.** Recortar el paso GN a Δ falló en Freudenstein & Roth: se detenía en f ≈ 32 (el mínimo es 24.49). Con J casi singular, ese paso puede bajar el modelo menos que Cauchy e incumplir la ec. 2.8 del artículo, que sostiene la convergencia. Solución: **usar el que más baje el modelo**.

## 6. Diferencias finitas (S3–S6): por qué DFO-GN existe

J se puede aproximar con f′ ≈ (f(x+h) − f(x))/h y h ≈ √ε ≈ 1.5·10⁻⁸ (equilibrio entre truncamiento y cancelación). Eso funciona sin ruido, pero **con ruido** σ el error es ≈ σ/h = 10⁻²/10⁻⁸ = **10⁶**: J es pura basura. Fue lo que medimos: 0 éxitos en las 2 100 corridas con ruido de la entrega final. DFO-GN interpola a escala Δ, mucho mayor que h, y el ruido pesa σ/Δ.

## 7. Radio inferior ρₖ (artículo, Algoritmo 1 y sección 4.5)

**El problema que resuelve.** Con ruido, un buen paso puede parecer malo (R < 0.1) y rechazarse. Sin protección, cada rechazo reduce Δ, y como el Jacobiano interpolado tiene un error del orden de σ/Δ, un Δ pequeño produce un J cada vez peor: el método termina por radio pequeño sin haber llegado.

**La regla.** Δ nunca baja de ρ. Si Δ ya está en ρ y hubo 3 rechazos seguidos, se reduce la cota: (ρ, Δ) ← (0.1ρ, 0.5ρ). Se empieza con ρ₀ = Δ₀ y se termina cuando ρ < 10⁻¹⁰. Son los valores del artículo (α₁ = 0.1, α₂ = 0.5).

**Ejemplo real.** Rosenbrock, ruido multiplicativo, semilla 2033: sin ρ, el radio cae hasta 5.5·10⁻¹¹ y el método se detiene en f = 0.46; con ρ, el radio nunca baja de 0.026 y llega a f ≈ 10⁻¹⁴ en 54 evaluaciones.

## 8. Simulación Monte Carlo (S25–S27)

**Qué se estima.** La probabilidad p de que un método alcance la tolerancia τ con ruido. Cada corrida da un indicador 1 (éxito) o 0, y la proporción p̂ = éxitos / corridas estima p (Ley de los Grandes Números).

**Incertidumbre.** Error estándar √(p̂(1−p̂)/n) e intervalo del 95 % p̂ ± 1.96·EE (Teorema Central del Límite). Con p̂ = 0 el intervalo normal tiene ancho cero: no es certeza, es un límite de la aproximación.

**Comparación por pares.** La corrida k usa la semilla 2026 + k en todos los métodos, así que todos reciben el mismo ruido. La diferencia dₖ = Iₖ(con ρ) − Iₖ(sin ρ) tiene intervalo d̄ ± 1.96·s/√n. Si no contiene el 0, la mejora no se debe al azar.

**Por qué 100 semillas.** El error estándar cae como 1/√n: con 10 semillas los intervalos de las dos versiones se traslapaban; con 100 se separan. Reducir el error a la mitad exige cuatro veces más corridas.

**Ruido uniforme por transformada inversa (S26).** Si U ~ U(0, 1), entonces e = a + (b−a)U ~ U(a, b). Con a = −√3σ y b = √3σ, la media es 0 y la varianza (b−a)²/12 = σ², igual que el ruido gaussiano.
