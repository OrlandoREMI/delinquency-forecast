"""
Contratos de datos para el pipeline de predicción.

crime_history_monthly — columnas requeridas:
    h3_index        str       índice H3 res=9
    año_mes         datetime  primer día del mes (ej. 2024-06-01)
    conteo          int       crímenes en esa celda ese mes
    zona_geografica str       "AMG" | "Interior"
    region          str       región administrativa de Jalisco
    clave_mun       int       clave INEGI del municipio
    municipio       str       nombre del municipio

crime_history_daily — columnas requeridas:
    h3_index        str       índice H3 res=9
    fecha           datetime  fecha del registro
    conteo          int       crímenes en esa celda ese día
    categoria       str       categoría de delito (opcional; requerida para near-repeat por categoría)

denue — columnas requeridas:
    h3_index        str
    poi_bancos      float
    poi_bares       float
    poi_escuelas    float
    poi_salud       float
    poi_conveniencia float
    poi_hoteles     float
    poi_gasolineras float
    poi_mercados    float
    poi_farmacias   float
    poi_total       float

inegi_inv — columnas requeridas:
    h3_index        str
    inv_banqueta_pct    float
    inv_alumpub_pct     float
    inv_drenajep_pct    float
    inv_transcol_pct    float
    inv_semapeat_pct    float
    inv_paratran_pct    float
    inv_rampas_pct      float
    inv_pasopeat_pct    float
    inv_ciclovia_pct    float
    inv_puestosf_pct    float
    inv_puestoam_pct    float
    inv_arboles_pct     float
    inv_guarnici_pct    float
    inv_n_segmentos     float
"""

CLASES = ["alto_impacto", "violencia_personal", "robo_confrontacion", "robo_patrimonial"]
CAT_SHORTS = ["alto", "viol", "conf", "patr"]

DELITO_CATEGORIA: dict[str, str] = {
    "Homicidio doloso":              "alto_impacto",
    "Feminicidio":                   "alto_impacto",
    "Violación":                     "alto_impacto",
    "Violencia familiar":            "violencia_personal",
    "Lesiones dolosas":              "violencia_personal",
    "Abuso sexual infantil":         "violencia_personal",
    "Robo a persona":                "robo_confrontacion",
    "Robo a negocio":                "robo_confrontacion",
    "Robo a cuentahabientes":        "robo_confrontacion",
    "Robo a bancos":                 "robo_confrontacion",
    "Robo a vehículos particulares": "robo_patrimonial",
    "Robo de motocicleta":           "robo_patrimonial",
    "Robo a int de vehículos":       "robo_patrimonial",
    "Robo de autopartes":            "robo_patrimonial",
    "Robo a casa habitación":        "robo_patrimonial",
    "Robo casa habitación":          "robo_patrimonial",
    "Robo a carga pesada":           "robo_patrimonial",
}

OUTPUT_COLS = [
    "h3_index", "municipio", "zona_geografica",
    "lambda_diario", "prob_crimen",
    "p_alto_impacto", "p_violencia_personal",
    "p_robo_confrontacion", "p_robo_patrimonial",
    "categoria_pred", "nivel_riesgo", "confiabilidad_score",
]

POI_COLS = [
    "poi_bancos", "poi_bares", "poi_escuelas", "poi_salud",
    "poi_conveniencia", "poi_hoteles", "poi_gasolineras",
    "poi_mercados", "poi_farmacias", "poi_total",
]

INV_COLS = [
    "inv_banqueta_pct", "inv_alumpub_pct", "inv_drenajep_pct",
    "inv_transcol_pct", "inv_semapeat_pct", "inv_paratran_pct",
    "inv_rampas_pct", "inv_pasopeat_pct", "inv_ciclovia_pct",
    "inv_puestosf_pct", "inv_puestoam_pct", "inv_arboles_pct",
    "inv_guarnici_pct", "inv_n_segmentos",
]

NR_CAT_COLS = [
    f"nr_cat_{s}_ring1_{w}d"
    for s in CAT_SHORTS
    for w in [7, 14]
]
