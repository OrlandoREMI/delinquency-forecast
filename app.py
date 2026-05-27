"""
Aplicación Gradio — Mapa de riesgo delictivo Jalisco.

Ejecutar:
    conda run -n geomod python app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import json
import time

import folium
import gradio as gr
import h3
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

import batch_predict as bp

# ---------------------------------------------------------------------------
# Constantes de UI
# ---------------------------------------------------------------------------
FEAT_PATH = ROOT / "data/processed/features_daily.parquet"
FECHA_MIN = "2017-01-01"
FECHA_DEFAULT = "2024-06-15"
TODAY    = pd.Timestamp.now().normalize()
DATA_END = pd.Timestamp("2025-12-30")

# Paleta semáforo 5 tonos (bajo → alto) y etiquetas correspondientes
PALETTE      = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
RISK_LABELS  = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]

CLASE_COLORS = {
    "alto_impacto":       "#c0392b",
    "violencia_personal": "#e67e22",
    "robo_confrontacion": "#2980b9",
    "robo_patrimonial":   "#27ae60",
}

CLASE_LABELS = {
    "alto_impacto":       "Alto impacto",
    "violencia_personal": "Violencia personal",
    "robo_confrontacion": "Robo confrontación",
    "robo_patrimonial":   "Robo patrimonial",
}

# Coordenadas por defecto (Guadalajara centro)
DEFAULT_LAT, DEFAULT_LON = 20.6736, -103.3440

# Máximo de celdas a renderizar en el mapa
MAX_CELLS_MAP = 1500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_geocoder = Nominatim(user_agent="delinquency-forecast-jalisco", timeout=10)


def geocode(query: str) -> tuple[float, float] | None:
    try:
        loc = _geocoder.geocode(f"{query}, Jalisco, Mexico")
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return None


def lam_to_risk(lam: float, vmin: float, vmax: float) -> tuple[str, str, int]:
    """Devuelve (color, etiqueta, score 1-99) derivados del mismo t,
    garantizando que color y etiqueta sean siempre consistentes."""
    if vmax <= vmin:
        return PALETTE[0], RISK_LABELS[0], 1
    t = max(0.0, min(1.0, (lam - vmin) / (vmax - vmin)))
    idx = min(int(t * len(PALETTE)), len(PALETTE) - 1)
    return PALETTE[idx], RISK_LABELS[idx], min(int(round(t * 100)), 99)


def composition_bar_html(row: pd.Series) -> str:
    clases = ["alto_impacto", "violencia_personal", "robo_confrontacion", "robo_patrimonial"]
    bars = ""
    for cls in clases:
        p = row.get(f"p_{cls}", 0) * 100
        color = CLASE_COLORS[cls]
        label = CLASE_LABELS[cls]
        bars += (
            f'<div title="{label}: {p:.1f}%" '
            f'style="display:inline-block;width:{p:.1f}%;height:10px;'
            f'background:{color};"></div>'
        )
    return f'<div style="width:200px;background:#ddd;border-radius:2px;overflow:hidden;">{bars}</div>'


def build_tooltip(row: pd.Series, risk_label: str, risk_score: int) -> str:
    return f"""
    <b>{row.get("municipio", "")}</b><br>
    <span style="font-size:11px;color:#555;">{row["h3_index"]}</span><br>
    <b>Índice de riesgo:</b> {risk_score}/99 &nbsp;—&nbsp; {risk_label}<br>
    <span style="font-size:10px;color:#777;">Relativo a celdas del día en esta vista &nbsp;·&nbsp; λ = {row["lambda_diario"]:.4f}</span><br>
    <b>Tipo probable:</b> {CLASE_LABELS.get(row["categoria_pred"], row["categoria_pred"])}<br>
    <b>Composición:</b><br>
    {composition_bar_html(row)}
    <div style="font-size:10px;margin-top:3px;">
      {"".join(
          f'<span style="color:{CLASE_COLORS[c]};margin-right:6px;">&#9632; {CLASE_LABELS[c]}: {row.get(f"p_{c}", 0)*100:.0f}%</span>'
          for c in ["alto_impacto","violencia_personal","robo_confrontacion","robo_patrimonial"]
      )}
    </div>
    <b>Confiabilidad:</b> {row["confiabilidad_score"]}/100
    """


def build_folium_map(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    zoom: int = 13,
) -> str:
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    lam_vals = df["lambda_diario"].values
    vmin = np.percentile(lam_vals, 10)
    vmax = np.percentile(lam_vals, 95)

    for _, row in df.iterrows():
        try:
            boundary = h3.cell_to_boundary(row["h3_index"])
            # h3 devuelve (lat, lon); GeoJSON necesita (lon, lat)
            coords = [[lon, lat] for lat, lon in boundary]
            coords.append(coords[0])  # cerrar polígono

            color, risk_label, risk_score = lam_to_risk(row["lambda_diario"], vmin, vmax)
            tooltip_html = build_tooltip(row, risk_label, risk_score)

            folium.GeoJson(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                },
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "color": "#888888",
                    "weight": 0.4,
                    "fillOpacity": 0.55,
                    "opacity": 0.4,
                },
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
            ).add_to(m)
        except Exception:
            continue

    # Leyenda
    legend_html = """
    <div style="
        position:fixed;bottom:30px;right:15px;z-index:1000;
        background:rgba(240,240,240,0.92);color:#222;
        padding:10px 14px;border-radius:8px;font-size:12px;
        border:1px solid #ccc;">
      <b>Índice de riesgo (1–99)</b><br>
      <span style="font-size:10px;color:#666;">Relativo a la vista actual</span><br>
      <div style="margin-top:6px;">
    """
    ranges = ["1–20", "20–40", "40–60", "60–80", "80–99"]
    for color, label, rng in zip(PALETTE, RISK_LABELS, ranges):
        legend_html += (
            f'<div><span style="background:{color};width:14px;height:14px;'
            f'display:inline-block;margin-right:6px;border-radius:2px;"></span>'
            f'{label} <span style="color:#666;font-size:10px;">({rng})</span></div>'
        )
    legend_html += """
      </div>
      <hr style="border-color:#bbb;margin:8px 0;">
      <b>Composición (hover)</b><br>
    """
    for cls in ["alto_impacto", "violencia_personal", "robo_confrontacion", "robo_patrimonial"]:
        legend_html += (
            f'<span style="color:{CLASE_COLORS[cls]};margin-right:4px;">&#9632;</span>'
            f'{CLASE_LABELS[cls]}<br>'
        )
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    return m._repr_html_()


# ---------------------------------------------------------------------------
# Función principal (llamada por Gradio)
# ---------------------------------------------------------------------------
def generar_mapa(fecha: str, busqueda: str, progreso=gr.Progress()):
    progreso(0, desc="Iniciando...")

    if not fecha:
        fecha = FECHA_DEFAULT
    busqueda = (busqueda or "").strip()

    # Verificar que el E3 exista
    e3_pkl = ROOT / "models/lgbm_e3_v1.pkl"
    if not e3_pkl.exists():
        return (
            "<div style='color:red;padding:20px;'>"
            "⚠️ El modelo E3 no está entrenado todavía.<br>"
            "Ejecuta primero:<br>"
            "<code>python src/crime_composition/03_train_e3.py</code>"
            "</div>",
            pd.DataFrame(),
        )

    progreso(0.15, desc="Cargando modelos y datos...")

    # Determinar filtro y centro del mapa
    center_lat, center_lon = DEFAULT_LAT, DEFAULT_LON
    zoom = 12
    h3_filter = None
    municipio_filter = None

    progreso(0.3, desc="Resolviendo búsqueda...")

    # Intentar detectar si es un H3 index directo
    if len(busqueda) == 15 and busqueda.startswith("8"):
        try:
            h3.cell_to_latlng(busqueda)
            lat, lon = h3.cell_to_latlng(busqueda)
            center_lat, center_lon = lat, lon
            zoom = 15
            # Mostrar celda + disco de radio 3
            h3_filter = list(h3.grid_disk(busqueda, 4))
        except Exception:
            pass

    if h3_filter is None and busqueda:
        # Intentar como nombre de municipio primero (rápido, sin geocodificar)
        municipio_filter = busqueda

        # Intentar geocodificar para centrar el mapa
        coords = geocode(busqueda)
        if coords:
            center_lat, center_lon = coords
            zoom = 13
        else:
            # Fallback: coordenadas de Guadalajara
            zoom = 12

    progreso(0.5, desc=f"Cargando predicciones para {fecha}...")

    result = bp.predict_batch(
        fecha=fecha,
        h3_indexes=h3_filter,
        municipio=municipio_filter,
    )

    if result is None or result.empty:
        return (
            "<div style='color:orange;padding:20px;'>"
            f"No hay datos para la fecha <b>{fecha}</b> o la búsqueda <b>{busqueda}</b>.<br>"
            f"Rango disponible: {FECHA_MIN} → {FECHA_MAX}."
            "</div>",
            pd.DataFrame(),
        )

    progreso(0.75, desc="Renderizando mapa...")

    # Limitar celdas para performance
    if len(result) > MAX_CELLS_MAP:
        result = result.nlargest(MAX_CELLS_MAP, "lambda_diario")

    # Banner para predicciones fuera del rango de datos reales
    ts = pd.Timestamp(fecha)
    days_from_data_end = max(0, (ts - DATA_END).days)
    if days_from_data_end > 180:
        future_banner = (
            f'<div style="background:#7c3aed;color:#fff;padding:8px 14px;'
            f'border-radius:6px;margin-bottom:8px;font-size:13px;">'
            f'⚠️ <b>Predicción especulativa</b> — {days_from_data_end} días después del último dato real (2025-12-30). '
            f'Lags y near-repeats reemplazados por promedios históricos del mes. Confiabilidad reducida.</div>'
        )
    elif days_from_data_end > 0:
        future_banner = (
            f'<div style="background:#d97706;color:#fff;padding:8px 14px;'
            f'border-radius:6px;margin-bottom:8px;font-size:13px;">'
            f'🔮 <b>Sin datos reales para esta fecha</b> — {days_from_data_end} días después del último dato (2025-12-30). '
            f'Lags y near-repeats reemplazados por promedios históricos del mes.</div>'
        )
    else:
        future_banner = ""

    # Si el centro no se pudo fijar con geocoder, usar centroide de las celdas
    if center_lat == DEFAULT_LAT and center_lon == DEFAULT_LON and len(result) > 0:
        lats, lons = [], []
        for h in result["h3_index"].head(200):
            try:
                la, lo = h3.cell_to_latlng(h)
                lats.append(la)
                lons.append(lo)
            except Exception:
                pass
        if lats:
            center_lat = float(np.mean(lats))
            center_lon = float(np.mean(lons))

    map_html = future_banner + build_folium_map(result, center_lat, center_lon, zoom)

    # Tabla resumen: top 20 por lambda_diario
    table = (
        result.nlargest(20, "lambda_diario")[
            [
                "municipio", "h3_index", "lambda_diario", "nivel_riesgo",
                "categoria_pred",
                "p_alto_impacto", "p_violencia_personal",
                "p_robo_confrontacion", "p_robo_patrimonial",
                "confiabilidad_score",
            ]
        ]
        .round({"lambda_diario": 4, "p_alto_impacto": 3, "p_violencia_personal": 3,
                "p_robo_confrontacion": 3, "p_robo_patrimonial": 3})
        .reset_index(drop=True)
    )

    progreso(1.0, desc="Listo.")
    return map_html, table


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="Forecasting Delictivo — Jalisco") as demo:
    gr.Markdown(
        """
        # 🗺️ Sistema de Predicción Delictiva — Jalisco
        Visualización de las **Etapas 2 y 3** del modelo de forecasting.
        Selecciona una fecha y busca un municipio, colonia o celda H3.
        """
    )

    with gr.Row():
        fecha_input = gr.Textbox(
            label="Fecha (YYYY-MM-DD)",
            value=FECHA_DEFAULT,
            placeholder="Ej: 2024-06-15",
            scale=1,
        )
        busqueda_input = gr.Textbox(
            label="Municipio / Colonia / Celda H3",
            value="Guadalajara",
            placeholder="Ej: Guadalajara  |  Zapopan  |  89483693653ffff",
            scale=3,
        )
        btn = gr.Button("Generar Mapa", variant="primary", scale=1)

    gr.Markdown(
        "_Datos históricos: 2017-01-01 → 2025-12-30 · "
        "Fechas futuras generan predicción especulativa con confiabilidad reducida · "
        "Máximo de celdas renderizadas: 1,500 (las de mayor riesgo)_"
    )

    mapa_out = gr.HTML(elem_id="mapa", label="Mapa de riesgo")

    gr.Markdown("### Top 20 celdas de mayor riesgo")
    tabla_out = gr.DataFrame(
        headers=[
            "Municipio", "H3", "λ diario", "Nivel",
            "Tipo probable",
            "P(alto impacto)", "P(violencia personal)",
            "P(robo conf.)", "P(robo patr.)",
            "Confiabilidad",
        ],
        label="Predicciones detalladas",
    )

    btn.click(
        fn=generar_mapa,
        inputs=[fecha_input, busqueda_input],
        outputs=[mapa_out, tabla_out],
    )

    # Ejecutar al cargar la página con valores por defecto
    demo.load(
        fn=generar_mapa,
        inputs=[fecha_input, busqueda_input],
        outputs=[mapa_out, tabla_out],
    )


if __name__ == "__main__":
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        prevent_thread_lock=True,
        theme=gr.themes.Base(),
        css="""
        .gradio-container { max-width: 1400px; margin: 0 auto; }
        #mapa { min-height: 600px; }
        """,
    )
    with open("/tmp/gradio_url.txt", "w") as f:
        f.write(f"local: {local_url}\npublic: {share_url}\n")
    print(f"Local:  {local_url}")
    print(f"Public: {share_url}")
    demo.block_thread()
