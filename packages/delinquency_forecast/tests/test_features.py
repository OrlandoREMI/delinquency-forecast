"""Tests unitarios de los módulos de features."""
import h3
import numpy as np
import pandas as pd
import pytest

from delinquency_forecast.features.calendar import date_features
from delinquency_forecast.features.nearrepeat import (
    build_nearrepeat_by_category,
    build_nearrepeat_global,
)
from delinquency_forecast.features.temporal import build_e1_features, build_e2_temporal
from delinquency_forecast.schemas import NR_CAT_COLS

from .conftest import _CELLS, _CENTER


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

class TestDateFeatures:
    def test_weekday(self):
        # 2023-11-27 es lunes
        f = date_features(pd.Timestamp("2023-11-27"))
        assert f["dia_semana"] == 0
        assert f["es_fin_semana"] == 0

    def test_weekend(self):
        # 2023-11-25 es sábado
        f = date_features(pd.Timestamp("2023-11-25"))
        assert f["dia_semana"] == 5
        assert f["es_fin_semana"] == 1

    def test_mexican_holiday(self):
        # 2023-11-20 — Revolución Mexicana
        f = date_features(pd.Timestamp("2023-11-20"))
        assert f["es_festivo"] == 1

    def test_regular_day_not_holiday(self):
        f = date_features(pd.Timestamp("2023-11-08"))
        assert f["es_festivo"] == 0

    def test_mes(self):
        f = date_features(pd.Timestamp("2023-06-15"))
        assert f["mes"] == 6


# ---------------------------------------------------------------------------
# nearrepeat
# ---------------------------------------------------------------------------

class TestNearRepeatGlobal:
    def _make_history(self, cell, n_days, count_per_day=1):
        dates = pd.date_range("2023-11-16", periods=n_days, freq="D")
        return pd.DataFrame({
            "h3_index": cell,
            "fecha": dates,
            "conteo": float(count_per_day),
        })

    def test_crimes_in_ring1_neighbor_counted(self):
        """Crímenes en vecino ring-1 deben aparecer en nr_ring1_7d."""
        neighbor = list(h3.grid_ring(_CENTER, 1))[0]
        history = self._make_history(neighbor, 7)
        target = pd.Timestamp("2023-11-23")

        result = build_nearrepeat_global([_CENTER], target, history)
        assert result.loc[_CENTER, "nr_ring1_7d"] == 7.0

    def test_crimes_in_ring2_not_in_ring1(self):
        """Vecino ring-2 no cuenta en nr_ring1_7d."""
        ring2_cell = list(h3.grid_ring(_CENTER, 2))[0]
        history = self._make_history(ring2_cell, 7)
        target = pd.Timestamp("2023-11-23")

        result = build_nearrepeat_global([_CENTER], target, history)
        assert result.loc[_CENTER, "nr_ring1_7d"] == 0.0
        assert result.loc[_CENTER, "nr_ring2_7d"] == 7.0

    def test_crimes_in_own_cell_not_counted(self):
        """La celda central no cuenta como vecino de sí misma."""
        history = self._make_history(_CENTER, 7)
        target = pd.Timestamp("2023-11-23")

        result = build_nearrepeat_global([_CENTER], target, history)
        assert result.loc[_CENTER, "nr_ring1_7d"] == 0.0

    def test_window_boundary(self):
        """Crímenes fuera de la ventana de 7 días no se cuentan."""
        neighbor = list(h3.grid_ring(_CENTER, 1))[0]
        # 10 días antes de target: está dentro de lag_14 pero fuera de lag_7
        history = pd.DataFrame({
            "h3_index": [neighbor],
            "fecha": [pd.Timestamp("2023-11-13")],
            "conteo": [5.0],
        })
        target = pd.Timestamp("2023-11-23")

        result = build_nearrepeat_global([_CENTER], target, history)
        assert result.loc[_CENTER, "nr_ring1_7d"] == 0.0
        assert result.loc[_CENTER, "nr_ring1_14d"] == 5.0

    def test_target_date_not_included(self):
        """Crímenes en la fecha objetivo no cuentan (solo días anteriores)."""
        neighbor = list(h3.grid_ring(_CENTER, 1))[0]
        target = pd.Timestamp("2023-11-23")
        history = pd.DataFrame({
            "h3_index": [neighbor],
            "fecha": [target],
            "conteo": [10.0],
        })
        result = build_nearrepeat_global([_CENTER], target, history)
        assert result.loc[_CENTER, "nr_ring1_1d"] == 0.0

    def test_empty_history_returns_zeros(self):
        empty = pd.DataFrame(columns=["h3_index", "fecha", "conteo"])
        target = pd.Timestamp("2023-11-23")
        result = build_nearrepeat_global([_CENTER], target, empty)
        assert (result == 0).all().all()

    def test_output_columns(self):
        empty = pd.DataFrame(columns=["h3_index", "fecha", "conteo"])
        result = build_nearrepeat_global([_CENTER], pd.Timestamp("2023-11-23"), empty)
        assert set(result.columns) == {"nr_ring1_1d", "nr_ring1_7d", "nr_ring1_14d", "nr_ring2_7d"}


class TestNearRepeatByCategory:
    def test_output_columns(self):
        empty = pd.DataFrame(columns=["h3_index", "fecha", "conteo", "categoria"])
        result = build_nearrepeat_by_category([_CENTER], pd.Timestamp("2023-11-23"), empty)
        assert set(result.columns) == set(NR_CAT_COLS)

    def test_no_categoria_column_returns_zeros(self):
        history = pd.DataFrame({
            "h3_index": [list(h3.grid_ring(_CENTER, 1))[0]],
            "fecha": [pd.Timestamp("2023-11-20")],
            "conteo": [5.0],
        })
        result = build_nearrepeat_by_category([_CENTER], pd.Timestamp("2023-11-23"), history)
        assert (result == 0).all().all()

    def test_category_specific_count(self):
        """Solo crímenes de la categoría correcta deben contarse."""
        neighbor = list(h3.grid_ring(_CENTER, 1))[0]
        history = pd.DataFrame({
            "h3_index": [neighbor, neighbor],
            "fecha": [pd.Timestamp("2023-11-20"), pd.Timestamp("2023-11-20")],
            "conteo": [3.0, 2.0],
            "categoria": ["alto_impacto", "robo_patrimonial"],
        })
        target = pd.Timestamp("2023-11-23")
        result = build_nearrepeat_by_category([_CENTER], target, history)

        assert result.loc[_CENTER, "nr_cat_alto_ring1_7d"] == 3.0
        assert result.loc[_CENTER, "nr_cat_patr_ring1_7d"] == 2.0
        assert result.loc[_CENTER, "nr_cat_viol_ring1_7d"] == 0.0


# ---------------------------------------------------------------------------
# temporal — E1
# ---------------------------------------------------------------------------

class TestBuildE1Features:
    @pytest.fixture
    def simple_history(self):
        months = pd.date_range("2022-01-01", periods=24, freq="MS")
        rows = [
            {"h3_index": _CENTER, "año_mes": m, "conteo": 4,
             "zona_geografica": "AMG", "region": "Centro",
             "clave_mun": 39, "municipio": "Guadalajara"}
            for m in months
        ]
        return pd.DataFrame(rows)

    @pytest.fixture
    def encoders(self):
        from delinquency_forecast import DelinquencyPipeline
        return DelinquencyPipeline.load()._e1.encoders

    def test_output_has_e1_feature_cols(self, simple_history, encoders):
        from delinquency_forecast import DelinquencyPipeline
        e1 = DelinquencyPipeline.load()._e1
        target = pd.Period("2023-12", freq="M")
        result = build_e1_features([_CENTER], target, simple_history, encoders)
        for col in e1.feature_cols:
            assert col in result.columns, f"Falta columna E1: {col}"

    def test_lag_1_matches_previous_month(self, simple_history, encoders):
        """lag_1 debe ser el conteo del mes anterior al target."""
        target = pd.Period("2023-12", freq="M")
        result = build_e1_features([_CENTER], target, simple_history, encoders)
        # El historial tiene conteo=4 en todos los meses
        assert result.loc[_CENTER, "lag_1"] == 4.0

    def test_future_period_uses_same_month_avg(self, simple_history, encoders):
        """Para un período futuro sin datos, el lag usa promedio histórico del mismo mes."""
        # Historial hasta 2023-12, target en 2025-06
        target = pd.Period("2025-06", freq="M")
        result = build_e1_features([_CENTER], target, simple_history, encoders)
        # lag_1 = 2025-05 (futuro) → promedio histórico de mayo = 4.0
        assert result.loc[_CENTER, "lag_1"] == pytest.approx(4.0, abs=0.5)

    def test_unknown_cell_returns_zeros(self, simple_history, encoders):
        unknown = h3.latlng_to_cell(19.0, -104.5, 9)
        target = pd.Period("2023-12", freq="M")
        result = build_e1_features([unknown], target, simple_history, encoders)
        assert result.loc[unknown, "lag_1"] == 0.0
        assert result.loc[unknown, "hist_max"] == 0.0


# ---------------------------------------------------------------------------
# temporal — E2
# ---------------------------------------------------------------------------

class TestBuildE2Temporal:
    @pytest.fixture
    def daily_history(self):
        dates = pd.date_range("2023-11-14", periods=14, freq="D")
        return pd.DataFrame({
            "h3_index": _CENTER,
            "fecha": dates,
            "conteo": 2.0,
        })

    def test_lag_1_correct(self, daily_history):
        target = pd.Timestamp("2023-11-28")
        result = build_e2_temporal([_CENTER], target, daily_history, {})
        # 2023-11-27 está en el historial con conteo=2
        assert result.loc[_CENTER, "lag_1"] == 2.0

    def test_lag_outside_history_is_zero(self, daily_history):
        target = pd.Timestamp("2023-11-28")
        result = build_e2_temporal([_CENTER], target, daily_history, {})
        # lag_14 = 2023-11-14, que está en el historial
        assert result.loc[_CENTER, "lag_14"] == 2.0

    def test_rolling_mean_7(self, daily_history):
        target = pd.Timestamp("2023-11-28")
        result = build_e2_temporal([_CENTER], target, daily_history, {})
        assert result.loc[_CENTER, "rolling_mean_7"] == pytest.approx(2.0, abs=0.1)

    def test_target_date_not_included_in_lags(self, daily_history):
        """El día objetivo no debe incluirse en lags ni rolling."""
        history_with_target = pd.concat([
            daily_history,
            pd.DataFrame({"h3_index": [_CENTER], "fecha": [pd.Timestamp("2023-11-28")], "conteo": [999.0]}),
        ])
        target = pd.Timestamp("2023-11-28")
        result = build_e2_temporal([_CENTER], target, history_with_target, {})
        assert result.loc[_CENTER, "lag_1"] == 2.0  # no 999

    def test_empty_history_returns_zeros(self):
        empty = pd.DataFrame(columns=["h3_index", "fecha", "conteo"])
        result = build_e2_temporal([_CENTER], pd.Timestamp("2023-11-28"), empty, {})
        assert result.loc[_CENTER, "lag_1"] == 0.0
        assert result.loc[_CENTER, "rolling_mean_7"] == 0.0
