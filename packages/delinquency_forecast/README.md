# delinquency-forecast

Pipeline de predicción espaciotemporal de incidencia delictiva para Jalisco. Dado el historial de crímenes por celda H3 (res=9), predice el conteo esperado para un día específico y la composición de riesgo por categoría de delito.

---

## Instalación

**Desde el repositorio:**
```bash
pip install "git+https://github.com/seigpol/delinquency-forecast.git#subdirectory=packages/delinquency_forecast"
```

---

## Uso básico

El paquete expone dos formas de obtener predicciones.

### Path 1 — on-the-fly

El pipeline recibe los datos crudos, construye los features internamente y corre los tres modelos en secuencia (E1 → E2 → E3). Es el path flexible: funciona para cualquier fecha, incluyendo fechas futuras.

```python
from delinquency_forecast import DelinquencyPipeline

pipeline = DelinquencyPipeline.load()

result = pipeline.predict(
    fecha="2024-06-15",
    crime_history_monthly=df_monthly,
    crime_history_daily=df_daily,
    denue=df_denue,
    inegi_inv=df_inegi,
    data_end="2024-12-31",   # opcional, pero recomendado
)
```

### Path 2 — features precomputadas

Si el backend ya tiene una tabla de features precomputadas, el pipeline solo corre E2 → E3. Es más rápido y pensado para el flujo operativo normal.

```python
result = pipeline.predict_from_features(
    fecha="2024-06-15",
    features_df=df_features,
    data_end="2024-12-31",
)
```

---

## Datos de entrada

### `crime_history_monthly` — historial mensual de crímenes

Tabla de conteos por celda H3 y mes. El pipeline usa los últimos 24 meses para construir los features de la Etapa 1 (lags, rolling stats, historial del mismo mes). Para fechas futuras que no tengan datos reales, los lags se reemplazan con el promedio histórico del mismo mes de calendario.

| Columna | Tipo | Descripción |
|---|---|---|
| `h3_index` | `str` | Índice H3 resolución 9 |
| `año_mes` | `datetime` | Primer día del mes (ej. `2024-06-01`) |
| `conteo` | `int` | Crímenes registrados en esa celda ese mes |
| `zona_geografica` | `str` | `"AMG"` o `"Interior"` |
| `region` | `str` | Región administrativa de Jalisco |
| `clave_mun` | `int` | Clave INEGI del municipio |
| `municipio` | `str` | Nombre del municipio |

### `crime_history_daily` — historial diario de crímenes

Tabla de conteos diarios por celda. El pipeline usa los 30 días previos a `fecha` para calcular lags diarios y features de near-repeat espacial. La columna `categoria` es opcional — si no está presente, las features de near-repeat por categoría se rellenan con cero, lo que reduce levemente la precisión de la Etapa 3.

| Columna | Tipo | Descripción |
|---|---|---|
| `h3_index` | `str` | Índice H3 resolución 9 |
| `fecha` | `datetime` | Fecha del registro |
| `conteo` | `int` | Crímenes registrados en esa celda ese día |
| `categoria` | `str` | `"alto_impacto"` / `"violencia_personal"` / `"robo_confrontacion"` / `"robo_patrimonial"` (nullable) |

### `denue` — puntos de interés por celda H3

Tabla estática.

| Columna | Tipo |
|---|---|
| `h3_index` | `str` |
| `poi_bancos` | `float` |
| `poi_bares` | `float` |
| `poi_escuelas` | `float` |
| `poi_salud` | `float` |
| `poi_conveniencia` | `float` |
| `poi_hoteles` | `float` |
| `poi_gasolineras` | `float` |
| `poi_mercados` | `float` |
| `poi_farmacias` | `float` |
| `poi_total` | `float` |

### `inegi_inv` — infraestructura urbana por celda H3

Tabla estática. Proporciones de infraestructura de calle derivadas del INEGI INV 2020.

| Columna | Tipo |
|---|---|
| `h3_index` | `str` |
| `inv_banqueta_pct` | `float` |
| `inv_alumpub_pct` | `float` |
| `inv_drenajep_pct` | `float` |
| `inv_transcol_pct` | `float` |
| `inv_semapeat_pct` | `float` |
| `inv_paratran_pct` | `float` |
| `inv_rampas_pct` | `float` |
| `inv_pasopeat_pct` | `float` |
| `inv_ciclovia_pct` | `float` |
| `inv_puestosf_pct` | `float` |
| `inv_puestoam_pct` | `float` |
| `inv_arboles_pct` | `float` |
| `inv_guarnici_pct` | `float` |
| `inv_n_segmentos` | `float` |

### `h3_indexes` (opcional)

Lista de índices H3 a predecir. Si no se pasa, el pipeline predice para todas las celdas presentes en `crime_history_monthly`. Útil para limitar la predicción a un municipio o zona específica sin tener que filtrar el historial antes de llamar al pipeline.

### `data_end` (opcional pero recomendado)

Fecha del último dato real disponible en el sistema (`'YYYY-MM-DD'`). El pipeline la usa para calcular el `confiabilidad_score` — cuanto más lejos esté `fecha` de `data_end`, mayor es la penalización. Si no se pasa, se infiere del máximo en `crime_history_monthly`, pero en el path de features precomputadas (`predict_from_features`) no hay forma de inferirlo, así que en ese caso conviene pasarlo explícitamente.

---

## PostgresLoader

Si el backend usa PostgreSQL, `PostgresLoader` construye y ejecuta las queries necesarias automáticamente. Solo necesita una conexión SQLAlchemy y que las tablas tengan los nombres y columnas esperados.

```python
from delinquency_forecast.loaders import PostgresLoader

loader = PostgresLoader(conn)  # SQLAlchemy Connection o Engine

result = pipeline.predict(
    fecha="2024-06-15",
    **loader.load(fecha="2024-06-15", municipio="Guadalajara")
)
```

`loader.load()` acepta `municipio` o `h3_indexes` (no los dos a la vez) y devuelve un dict listo para desempacar en `pipeline.predict()`, incluyendo `data_end` inferido del máximo `fecha` en `crime_daily`.

### Tablas esperadas en la base de datos

| Tabla | Columnas mínimas |
|---|---|
| `crime_monthly` | `h3_index, año_mes, conteo, municipio, zona_geografica, region, clave_mun` |
| `crime_daily` | `h3_index, fecha, conteo, municipio, categoria` |
| `denue` | `h3_index, poi_bancos, ..., poi_total` |
| `inegi_inv` | `h3_index, inv_banqueta_pct, ..., inv_n_segmentos` |

Si los nombres de las tablas son distintos, se pueden sobreescribir en el constructor:

```python
loader = PostgresLoader(conn, tables={
    "crime_monthly": "crimen_mensual",
    "crime_daily":   "crimen_diario",
})
```

---

## Output

`pipeline.predict()` y `pipeline.predict_from_features()` devuelven un `pd.DataFrame` con una fila por celda H3 y las siguientes columnas:

| Columna | Tipo | Descripción |
|---|---|---|
| `h3_index` | `str` | Índice H3 resolución 9 |
| `municipio` | `str` | Nombre del municipio |
| `zona_geografica` | `str` | `"AMG"` o `"Interior"` |
| `lambda_diario` | `float64` | Crímenes esperados ese día (Poisson λ, salida E2) |
| `prob_crimen` | `float64` | Probabilidad de ≥1 crimen en % → `(1 − e^−λ) × 100` |
| `p_alto_impacto` | `float64` | Prob. de que el crimen sea de alto impacto (E3 calibrado) |
| `p_violencia_personal` | `float64` | Prob. violencia personal |
| `p_robo_confrontacion` | `float64` | Prob. robo con confrontación |
| `p_robo_patrimonial` | `float64` | Prob. robo patrimonial |
| `categoria_pred` | `str` | Categoría más probable según threshold ajustado |
| `nivel_riesgo` | `str` | `"muy_bajo"` / `"bajo"` / `"medio"` / `"alto"` / `"muy_alto"` |
| `color_riesgo` | `str` | Color hex asociado al nivel — listo para renderizar en mapa |
| `confiabilidad_score` | `int64` | 0–100, penaliza fechas fuera del historial real |

### Aclaraciones

**`nivel_riesgo` y `color_riesgo` son relativos al batch.** Se calculan por quintiles de λ dentro del conjunto de celdas que se pasaron en la consulta. Si pides solo Guadalajara, "alto" significa alto relativo a Guadalajara, no a Jalisco completo. La paleta es `#1a9641 → #a6d96a → #ffffbf → #fdae61 → #d7191c` (verde → rojo).
> TODO: Implementar riesgo relativo al histórico y/o estatal

**Las columnas `p_*` suman 1.0.** Representan la composición de riesgo condicional a que ocurra un crimen — no la probabilidad absoluta de cada tipo. Para la probabilidad absoluta de que ocurra cualquier crimen usa `prob_crimen`.

**Fechas futuras.** El pipeline las acepta sin problema. Los features de lags y near-repeat que caen fuera del historial real se reemplazan con promedios históricos del mismo mes. El `confiabilidad_score` baja automáticamente según qué tan lejos esté la fecha de `data_end`.
