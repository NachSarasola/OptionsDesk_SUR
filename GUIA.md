# Guia de uso — OptionsDesk GGAL

## Que hace este bot

El bot analiza todas las opciones disponibles de GGAL en BYMA, calcula la tasa anualizada que podes cobrar con dos estrategias (lanzamiento cubierto y venta de put), y te muestra **la mejor jugada para cada perfil de riesgo**.

No necesitas saber teoria de opciones. Solo lees la tarjeta verde, aprietas "Generar ticket" y pegas la orden en Bull Market.

---

## Como usarlo (5 pasos)

1. **Abre el dashboard** con `streamlit run optionsdesk/ui/dashboard.py`
2. **Mira la tab "Inicio"** — ves tres tarjetas: Conservadora, Equilibrada y Agresiva
3. **Elige la tarjeta que te convence** — el semaforo verde significa que paso todos los filtros de riesgo
4. **Haz clic en "Generar ticket"** — te aparece el texto exacto de la orden para copiar
5. **Pega la orden en Bull Market** — ejecutas vos manualmente; el bot no opera solo

**Sidebar:**
- *Capital disponible*: ingresa cuanto tenes en pesos → el bot calcula cuantos lotes entras y la ganancia estimada
- *Modo avanzado*: activa la cadena completa, el simulador de P&L y el historial para explorar mas
- *Contexto de mercado*: modulo experimental; mejora con el tiempo a medida que el recorder acumula historial

---

## Lanzamiento cubierto

**Que es:** Compras acciones de GGAL y simultáneamente vendes el derecho a que otra persona te las compre a un precio fijo (el strike).

**Como ganas plata:** Al vender ese derecho cobras una prima. Esa prima es tu ganancia asegurada si GGAL cierra arriba del strike al vencimiento. Si baja, la prima te amortigua la caida.

**Ejemplo simplificado:**
- GGAL a $8.500. Vendes call K=$9.000 a $200 de prima.
- Si GGAL termina encima de $9.000: te "ejercen", venden tu accion a $9.000. Ganaste la prima + la suba hasta $9.000.
- Si GGAL baja: perdiste en la accion, pero la prima de $200 te cubre hasta $8.300. Ese es tu colchon.

**Que puede salir mal:** si GGAL cae mucho mas del colchon, perdes en la accion. No es garantia de rentabilidad — es renta con seguro parcial.

---

## Venta de put (cash-secured)

**Que es:** Le vendés a alguien el derecho de venderte acciones de GGAL a un precio fijo. A cambio cobras una prima ahora.

**Como ganas plata:** Si GGAL se mantiene arriba del strike al vencimiento, el put expira sin valor y te quedas con toda la prima. Esa prima es tu tasa.

**Ejemplo simplificado:**
- GGAL a $8.500. Vendes put K=$8.000 a $300 de prima.
- Si GGAL termina encima de $8.000: el put no se ejerce, guardas los $300.
- Si GGAL cae a $7.500: te obligan a comprar a $8.000. Pagaste $8.000 pero ya tenias $300 de prima → costo real $7.700. Perdes si cae mas.

**Que puede salir mal:** si GGAL cae fuerte, terminas comprando acciones a un precio por encima del mercado. La prima cubre solo parcialmente.

---

## Holdear vs Swing: cuando cerrar antes

El bot analiza automaticamente si conviene esperar al vencimiento o cerrar la posicion antes. Para cada recomendacion vas a ver un badge:

- **HOLD** — el modelo dice que aguantar hasta el vencimiento maximiza tu retorno ajustado por riesgo.
- **SWING** — conviene cerrar en ~N dias, capturando X% de la ganancia total. Despues de ese punto, el gamma (riesgo de movimiento brusco) crece mas rapido que la theta (ganancia por tiempo) que queda.

**Por que no siempre es mejor aguantar:** La prima de una opcion no decae de manera lineal. Los ultimos dias aportan theta acelerada pero tambien gamma desproporcionado: si el precio se mueve, la opcion puede recuperar mucho valor rapido. El bot cuantifica ese trade-off y elige el dia optimo de salida para tu perfil de riesgo.

**Como ejecutar el swing:**
1. El badge dice "SWING — cerra en ~12 dias".
2. Cuando llega ese dia (o antes si el mercado se mueve a tu favor), entras a Bull Market.
3. Buscas la posicion y ejecutas la recompra de la opcion al precio de mercado.
4. Si configuraste el monitor (`HORIZON_MONITOR_ENABLED=true`), recibis una alerta de Telegram en el momento exacto en que se alcanza el % de captura objetivo.

---

## Analisis tecnico e ideas direccionales

El bot incluye una tab **Direccional** con analisis tecnico real sobre el precio diario de GGAL (fuente: BYMA Open Data via PyOBD, ~20 min de delay).

**Que muestra:**
- Grafico de precio con SMA(5) y SMA(20)
- RSI(14) con niveles de sobrecompra/sobreventa
- ATR(14) como medida de volatilidad real del mercado
- Momentum a 10 dias
- Veredicto de tendencia: ALCISTA / BAJISTA / LATERAL

**Ideas direccionales:**
Cuando la señal tecnica es lo suficientemente fuerte (confianza media+, momentum >1.5%), el bot sugiere una idea de compra de call o put. Estas ideas son **especulativas**: el bot no tiene un modelo de alpha validado. La logica es simple — AT estandar sobre el subyacente — y la perdida maxima esta limitada a la prima pagada.

Las ideas son swing de dias, no scalping. El ATR se usa para calcular objetivo (2x ATR) y stop (1x ATR).

**Por que esta en una tab separada:** El carry (cobrar prima) es el edge real del bot. Lo direccional es especulativo y se mantiene separado visualmente para que quede claro.

---

## Contexto multi-timeframe (HTF / LTF)

Un analisis de mercado serio sigue un enfoque "top-down": primero el marco alto define el sesgo, despues el diario da el setup, y el intradiario la confirmacion.

El bot analiza GGAL en tres marcos:

| Marco | Frecuencia | Que define |
|---|---|---|
| **HTF (Semanal)** | Velas semanales | Sesgo dominante de largo plazo |
| **Diario (base)** | Velas diarias | Setup: donde esta el precio hoy |
| **LTF (Intradiario)** | Barras de 1 min (last 60) | Confirmacion o timing de entrada |

**Badges de alineacion:**
- **Marcos alineados alcistas** (verde): semanal y diario apuntan para arriba → señal fuerte
- **Marcos alineados bajistas** (rojo): semanal y diario apuntan para abajo → señal fuerte en contra
- **Conflicto HTF / Diario** (naranja): los marcos se contradicen → el bot es mas cauteloso con ideas especulativas
- **Marcos neutros** (azul): lateral en alguno o ambos marcos

**Regimen de volatilidad:**
- **Vol en expansion** (naranja): el ATR reciente supera 125% del ATR promedio → el mercado esta movido; opciones mas caras
- **Vol en contraccion** (azul): el ATR reciente esta por debajo del 75% del promedio → mercado quieto; opciones mas baratas
- **Vol normal**: ATR dentro de rango historico normal

**Como afecta las recomendaciones:** El contexto MTF *informa* pero nunca *elimina* recomendaciones de carry. El carry (cobrar prima) es el edge validado. La alineacion de marcos ajusta levemente el score: si el semanal y el diario estan alineados en la misma direccion que la estrategia, el score sube un poco; si hay conflicto, baja y aparece una advertencia.

---

## Spreads verticales — riesgo definido

Un spread vertical es una estrategia de 2 patas: compras una opcion y vendes otra del mismo tipo (ambas calls o ambas puts), mismo vencimiento, distinto strike.

**Por que son mejores que la opcion desnuda para ideas especulativas:**
- **Riesgo definido**: no podes perder mas del debito que pagaste (o del ancho del spread menos el credito cobrado)
- **Costo menor**: vender una pata financia parte de la compra → entrada mas barata
- **Disciplina**: el tope de ganancia te obliga a ser preciso en la seleccion de strikes

El trade-off: la ganancia maxima tambien esta limitada. Es un intercambio consciente: menos costo, menos riesgo, pero menos potencial de suba.

### Los 4 tipos de spread vertical

**Bull Call Spread** (alcista, debito):
- Compras call K bajo + vendes call K alto
- Ganas si GGAL sube por encima del break-even
- Perdida max = debito pagado; Ganancia max = (ancho - debito) × 100

**Bear Call Spread** (bajista, credito):
- Vendes call K bajo + compras call K alto
- Cobras un credito hoy; ganas si GGAL se queda debajo del strike vendido
- Ganancia max = credito cobrado × 100; Perdida max = (ancho - credito) × 100

**Bull Put Spread** (alcista, credito):
- Vendes put K alto + compras put K bajo
- Cobras un credito hoy; ganas si GGAL se queda arriba del break-even
- Ganancia max = credito cobrado × 100; Perdida max = (ancho - credito) × 100

**Bear Put Spread** (bajista, debito):
- Compras put K alto + vendes put K bajo
- Ganas si GGAL cae por debajo del break-even
- Perdida max = debito pagado; Ganancia max = (ancho - debito) × 100

### Metricas del spread (v2.4 — nucleo cuant honesto)

| Metrica | Que significa |
|---|---|
| **Debito / Credito neto** | Lo que pagas (debito) o cobras (credito) al armar el spread, a precios ejecutables |
| **Ancho del spread** | Diferencia entre los dos strikes |
| **Break-even** | Precio de GGAL al vencimiento donde empatas |
| **R:R (Risk/Reward)** | Max ganancia / Max perdida. Solo descriptivo — no es el criterio de ranking |
| **PoP (fisica)** | Probabilidad calculada con **volatilidad realizada** (no IV) y **drift 0** (no la tasa de caucion). Es la PoP real bajo la distribucion historica del precio |
| **EV (valor esperado)** | E[payoff] en ARS por contrato. Integra el payoff del spread contra la distribucion fisica de S_T. EV > 0 = el spread tiene edge esperado |
| **EV / riesgo** | EV dividido por la perdida maxima, en %. Es el criterio de ranking: el spread con mayor retorno esperado sobre capital arriesgado sale primero |

**Por que importa la diferencia entre PoP fisica y risk-neutral:**
La PoP risk-neutral usa la IV (inflada por el premio de riesgo de volatilidad) y la tasa de caucion como drift (~53% anual en Argentina). Eso genera sesgo alcista enorme y sobreestima la probabilidad de ganancia. La PoP fisica usa la vol realizada y drift 0, que es conservadora pero honesta.

**Precios ejecutables:**
El bot pricetea las patas a precios de ejecucion reales: compra al ask, vende al bid. Un spread que parece rentable a mid puede no serlo al ejecutar. Con dos patas, los dos cruces del spread bid-ask se acumulan. Si el EV sigue siendo positivo a esos precios, el edge es real.

### Metricas adicionales del spread (v2.5 / v2.6 / v2.7)

| Metrica | Que significa |
|---|---|
| **Greeks netos** | `net_delta` = exposicion direccional (long − short × 100). `net_theta` = ARS/dia: credit spread positivo (ganas tiempo), debit spread negativo (sangras). `net_vega` = impacto de 1pp de Δvol: credit < 0 (short vol), debit > 0 (long vol). |
| **P(toque K)** | Probabilidad de que el spot toque el strike corto *en algun momento* durante la vida del spread (principio de reflexion). Siempre ≥ PoP terminal. Para opciones americanas (BYMA) es el riesgo real de asignacion temprana. Umbral: >40% = alto, 20-40% = moderado. |
| **Vol rolling 20d vs historica** | La vol 20d captura el regimen reciente; la serie completa lo suaviza. Se usa la ventana 20d para EV/PoP/VRP. Cuando el regimen cambia abruptamente, la 20d lo refleja antes. |
| **VRP honesto** | IV del short leg via solver Brent (CRR americano) vs vol realizada 20d. VRP > 0: vendes vol cara = edge real. VRP < 0: vendes vol barata = edge negativo. Antes se usaba un fallback de IV sin resolver. |
| **Yang-Zhang (YZ)** | Estimador de vol OHLCV ~7× mas eficiente que close-to-close. Captura gaps overnight y movimientos open-to-close. Se usa automaticamente como estimador primario cuando el historial tiene columnas OHLCV (PyOBD diario). Si no hay OHLCV, cae a C-C rolling 20d. El badge en la tab Direccional muestra "YZ-20d" o "C-C-20d". |

### Advertencias importantes

- **Legging risk**: el spread se arma con dos ordenes separadas en BYMA. El precio puede moverse entre la primera y la segunda pata → ejecutar lo mas rapido posible o usar ordenes limit agresivas
- **Spreads bid-ask acumulados**: dos patas = dos spreads bid-ask. Con spreads amplios (>30%), el edge se reduce drasticamente. El bot filtra patas poco liquidas
- **Liquidez de puts de GGAL**: historicamente pobre en BYMA. Si el bot no muestra spreads de puts, es por falta de liquidez — no por error
- **EV usa vol realizada backward-looking**: si el regimen de volatilidad cambia abruptamente, la estimacion puede ser optimista o pesimista. Chequeá el badge "Vol en expansion/contraccion" en los badges MTF

---

## Por que el bot no hace day-trading direccional

Pregunta frecuente: "Por que el bot no me dice 'compra una call porque va a subir'?"

Tres razones tecnicas concretas:

1. **Datos**: el recorder snapshotea cada 120 segundos. Granularidad insuficiente para scalping; no hay feed de ticks en tiempo real.
2. **Liquidez**: las opciones de GGAL tienen spreads bid-ask de hasta 30%. Cruzar el spread dos veces en el mismo dia destruye cualquier edge aparente.
3. **Edge**: el bot tiene un edge demostrable en el carry (cobrar prima mas cara que la volatilidad realizada). No tiene un modelo de alpha direccional validado — apostar a que "sube porque tiene momentum" sin backtesting riguroso es especular, no operar.

Lo que si hacemos: entrada a escala dias + monitoreo intradiario para salir con precision en el momento optimo. Eso te da rotacion activa de capital sin el casino intradiario.

---

## Glosario

| Termino | Que significa |
|---|---|
| **TNA** | Tasa Nominal Anual — el rendimiento anualizado de la estrategia. Para comparar con la caucion. |
| **Spread vs caucion** | Cuanto le ganas a la caucion colocadora (el deposito "sin riesgo" de Argentina). Un spread de +20% significa que la opcion te da 20 puntos porcentuales mas que dejar la plata en la caucion. |
| **Colchon** | Cuanto puede caer GGAL antes de que pierdas plata. Si el colchon es 8%, GGAL puede bajar 8% y todavia empatas. |
| **Probabilidad** | La chance de que la estrategia salga como se proyecta — derivada del delta de la opcion (calculado con CRR binomial). |
| **Semaforo verde** | La oportunidad paso todos los filtros: riesgo aprobado, probabilidad mayor a 65% y score mayor a 60/100. |
| **Semaforo amarillo** | Recomendable pero con advertencias — revisa el detalle. |
| **Score** | Puntaje de 0 a 100 que combina spread, colchon, probabilidad y liquidez segun el perfil. |
| **Delta** | Sensibilidad del precio de la opcion al movimiento de la accion. Se usa como proxy de probabilidad. |
| **Lotes** | Un contrato de opciones = 100 acciones. Si el bot dice "3 contratos" son 300 acciones. |
| **Theta** | Cuanto vale menos la opcion por cada dia que pasa. Para el vendedor es ganancia: cada dia que pasa sin movimiento cobra un poco mas de la prima. |
| **Gamma** | Cuanto cambia el delta cuando se mueve el spot. Gamma alto = la opcion puede recuperar valor rapido si el mercado se mueve. Riesgo para el vendedor. |
| **Vega** | Sensibilidad de la opcion a cambios en la volatilidad implicita. Vega alto = si la vol sube, la opcion vale mas (malo para el vendedor). |
| **IV (Volatilidad implicita)** | Cuanta volatilidad futura "descuenta" el mercado en el precio de la opcion. Si IV > vol realizada = estas cobrando cara la prima. |
| **VRP (Variance Risk Premium)** | IV menos volatilidad realizada. Positivo = edge del vendedor: el mercado paga mas por proteccion de la que estadisticamente deberia. |
| **IV Rank** | Posicion de la IV actual en su rango historico (0-100). IV Rank alto = la prima esta cara respecto a la historia. Favorece al vendedor. |
| **HOLD / SWING** | HOLD = aguantar al vencimiento. SWING = cerrar antes del vencimiento para capturar la ganancia optima y liberar capital. |
| **Captura %** | Porcentaje de la ganancia total (hold hasta vencimiento) que ya fue capturada al cerrar el swing. |
| **SMA** | Simple Moving Average (media movil simple). SMA(5) = promedio de los ultimos 5 dias. Cruce SMA5 > SMA20 = señal alcista. |
| **RSI** | Relative Strength Index. Oscila entre 0 y 100. >70 = sobrecomprado (posible corrección). <30 = sobrevendido (posible rebote). 50 = neutral. |
| **ATR** | Average True Range. Mide la volatilidad real del mercado en pesos por dia. ATR alto = movimientos bruscos. Util para dimensionar stops y objetivos. |
| **Momentum** | Cambio porcentual del precio respecto a N dias atras. Positivo = el precio sube. Negativo = el precio baja. |
| **BUY_CALL / BUY_PUT** | Idea direccional especulativa: comprar una call (apuesta alcista) o una put (apuesta bajista). Perdida maxima = prima pagada. |
| **SMC** | Smart Money Concepts — metodologia de analisis de estructura de mercado institucional (grandes jugadores). |
| **BOS** | Break of Structure — el precio rompe el ultimo maximo/minimo significativo, confirmando la tendencia. BOS_UP = ruptura alcista; BOS_DOWN = bajista. |
| **CHoCH** | Change of Character — el precio rompe en direccion contraria a la tendencia dominante. Señal de posible cambio de tendencia; el bot lo usa para reducir confianza en la idea direccional. |
| **FVG** | Fair Value Gap — zona de desequilibrio entre tres velas consecutivas que el precio no ha llenado. Actua como iman: el precio tiende a volver a ese rango. Se usa como objetivo de precio (target). |
| **OB** | Order Block — ultima vela opositora antes de un BOS. Representa zona de acumulacion o distribucion institucional; actua como soporte/resistencia fuerte. Se usa como nivel de stop. |
| **HTF** | Higher Time Frame — marco temporal alto. En el bot: velas semanales. Define el sesgo dominante de largo plazo. |
| **LTF** | Lower Time Frame — marco temporal bajo. En el bot: barras de 1 minuto (intradiario). Confirmacion de entrada. |
| **MTF Alignment** | Alineacion multi-timeframe. Cuando semanal y diario apuntan en la misma direccion, la señal es mas confiable. |
| **Vol Regime** | Regimen de volatilidad actual respecto al historico. Expansion = opciones caras; Contraccion = opciones baratas. |
| **Bull Call Spread** | Spread alcista de debito: BUY call K_bajo + SELL call K_alto. Gana si el subyacente sube. |
| **Bear Call Spread** | Spread bajista de credito: SELL call K_bajo + BUY call K_alto. Gana si el subyacente baja o se queda quieto. |
| **Bull Put Spread** | Spread alcista de credito: SELL put K_alto + BUY put K_bajo. Gana si el subyacente sube o se queda quieto. |
| **Bear Put Spread** | Spread bajista de debito: BUY put K_alto + SELL put K_bajo. Gana si el subyacente baja. |
| **R:R** | Risk/Reward ratio. Cuanto ganas por cada peso que arriesgas. R:R = 2x → ganas $2 por cada $1 en riesgo. |
| **PoP** | Probability of Profit. Probabilidad risk-neutral (BSM) de que el spread resulte ganador al vencimiento. |
| **Debito neto** | Monto neto que se paga al armar un spread de debito (gasto). Perdida maxima = debito × 100. |
| **Credito neto** | Monto neto que se cobra al armar un spread de credito (ingreso). Ganancia maxima = credito × 100. |

---

## Edge cuantitativo: VRP, EV y PoP (v2.4)

El bot ahora rankea las oportunidades por **valor esperado** (EV), no por TNA bruta.

### Por que importa el VRP

El **Variance Risk Premium** (VRP = IV − vol realizada) es el edge estructural del vendedor de prima. Cuando IV > vol realizada, la opcion que vendiste esta sobrevaluada en promedio: la diferencia entre lo que cobra el mercado y lo que la accion efectivamente mueve es tu ganancia esperada.

Formalmente: `EV = prima_cobrada × P(favorable) − E[pérdida]`. Por construccion, `EV > 0 ⟺ VRP > 0`. Rankear por EV es rankear por el edge real del trade — no por el premio nominal.

### Badges en las recomendaciones

Cada tarjeta muestra tres metricas adicionales:

| Badge | Que significa |
|-------|--------------|
| **Vol edge** | IV rica / neutra / barata. "IV rica" = VRP positivo = mejor momento para vender. |
| **PoP fisica** | Probabilidad de ganancia calculada con la vol *realizada* (no la IV). Mas honesta que el delta: usa el breakeven real y la distribucion historica del subyacente. |
| **EV anual** | Expectativa anualizada como % del capital comprometido. EV > 0 = edge positivo. El numero en ARS es por lote (100 acciones). |

### Bandas de delta por perfil

El bot da un bonus de score cuando el delta de la opcion escrita cae en el rango objetivo:

| Perfil | Delta objetivo |
|--------|---------------|
| CONSERVADOR | 0.15 – 0.30 (OTM lejana, alta PoP) |
| EQUILIBRADO | 0.25 – 0.40 (OTM moderada) |
| AGRESIVO | 0.35 – 0.50 (casi ATM, max VRP) |

### Penalizacion de eventos

Cuando el vencimiento de la opcion cruza un evento conocido (earnings de GGAL, reunion BCRA), el score baja 8 puntos y aparece un aviso en la tarjeta. No es un gate — un evento no siempre es malo para el vendedor — pero si hay mayor incertidumbre (gap risk + IV crush post-evento).

---

## Gestion activa de posiciones (v2.4)

El monitor ahora evalua cuatro posibles acciones, en orden de prioridad:

| Señal | Condicion | Accion sugerida |
|-------|-----------|----------------|
| **TAKE_PROFIT** | Captura >= objetivo (default 50%) | Cerrar. Liberas capital antes del gamma de fin de ciclo. |
| **STOP** | Opcion vale >= 2× la prima cobrada (perdes >= 1× prima) | Cerrar. La perdida ya supero el limite prefijado. |
| **ROLL** | Quedan <= 21 dias a vencimiento y la opcion tiene valor | Rolar al proximo vencimiento con delta similar. |
| **DEFEND** | Delta de la opcion escrita >= 0.50 | El strike esta siendo testeado. Revisar o cubrir. |

Podes personalizar los parametros por posicion agregando al JSONL:
```json
{ ...campos_existentes..., "max_loss_mult": 3.0, "roll_dte": 14, "defend_delta": 0.45 }
```
Si no los pones, usa los defaults (2.0 / 21 / 0.50).

---

## Scanner de oportunidades intraday (v2.4, experimental)

El tab **Oportunidades** tiene un panel "Intraday — spikes de prima" que detecta:

- **WRITE** — la prima subio mas de lo que explica el delta (IV subio) → ventana para lanzar
- **CLOSE** — una posicion corta tuya bajo > 30% en los ultimos minutos → capturaste rapido

### Como funciona

El bot lee los snapshots del recorder (~120s de granularidad) y descompone cada cambio de prima en:

```
Δprima_total = Δprima_delta + Δprima_IV
Δprima_delta ≈ delta × Δspot           ← movimiento del subyacente (no es edge)
Δprima_IV    = residual                ← lo que no explica el spot = IV moviendose
```

Un z-score alto del residual IV = prima se encarecio mas alla de lo que justifica el subyacente → buena ventana para vender.

### Limitaciones honestas

- BYMA Open Data demora **~15-20 min** → no es scalping milisegundos, es swing intradiario.
- La liquidez de opciones GGAL intradiaria es **fina** → un unico print puede generar una falsa señal. Siempre verificar bid-ask antes de operar.
- Solo funciona si el **recorder esta corriendo** ese dia.

---

## Nota importante

> El bot te muestra la mejor jugada segun criterios quant, pero **no opera solo**. Vos sos quien decide ejecutar en Bull Market.
>
> El semaforo verde **reduce** el riesgo, no lo elimina. Las opciones tienen riesgo de mercado real: si GGAL cae mucho mas del colchon, perdes dinero. Operar con capital que puedas permitirte inmovilizar por el plazo del vencimiento.
>
> Los datos en modo demo son sinteticos. Para operar con datos reales, configura las credenciales en `.env`.
