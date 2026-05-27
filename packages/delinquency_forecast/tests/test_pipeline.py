"""Tests de integración del pipeline completo."""
import h3
import numpy as np
import pandas as pd
import pytest

from delinquency_forecast.schemas import CLASES, OUTPUT_COLS

from .conftest import FUTURE_DATE, HIST_DATE


class TestPredictOnTheFly:
    """Path completo: E1 → features → E2 → E3."""

    @pytest.fixture(autouse=True)
    def _result(self, pipeline, h3_cells, crime_history_monthly, crime_history_daily, denue, inegi_inv):
        self.result = pipeline.predict(
            HIST_DATE,
            crime_history_monthly,
            crime_history_daily,
            denue,
            inegi_inv,
        )

    def test_output_columns(self):
        assert list(self.result.columns) == OUTPUT_COLS

    def test_row_count_matches_cells(self, h3_cells):
        assert len(self.result) == len(h3_cells)

    def test_lambda_non_negative(self):
        assert (self.result["lambda_diario"] >= 0).all()

    def test_prob_crimen_in_range(self):
        assert self.result["prob_crimen"].between(0, 100).all()

    def test_composition_probs_sum_to_one(self):
        prob_cols = [f"p_{c}" for c in CLASES]
        sums = self.result[prob_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_composition_probs_non_negative(self):
        for c in CLASES:
            assert (self.result[f"p_{c}"] >= 0).all()

    def test_categoria_pred_valid(self):
        assert self.result["categoria_pred"].isin(CLASES).all()

    def test_nivel_riesgo_valid(self):
        assert self.result["nivel_riesgo"].isin(["bajo", "medio", "alto"]).all()

    def test_confidence_score_range(self):
        assert self.result["confiabilidad_score"].between(0, 100).all()

    def test_h3_indexes_in_output(self, h3_cells):
        assert set(self.result["h3_index"]) == set(h3_cells)


class TestPredictFutureDate:
    """Fechas más allá del historial activan fallback con promedios históricos."""

    @pytest.fixture(autouse=True)
    def _results(self, pipeline, crime_history_monthly, crime_history_daily, denue, inegi_inv):
        self.hist = pipeline.predict(
            HIST_DATE, crime_history_monthly, crime_history_daily, denue, inegi_inv
        )
        self.future = pipeline.predict(
            FUTURE_DATE, crime_history_monthly, crime_history_daily, denue, inegi_inv
        )

    def test_future_does_not_crash(self):
        assert len(self.future) > 0

    def test_future_output_columns(self):
        assert list(self.future.columns) == OUTPUT_COLS

    def test_future_lambda_non_negative(self):
        assert (self.future["lambda_diario"] >= 0).all()

    def test_future_lower_confidence_than_historical(self):
        assert self.future["confiabilidad_score"].mean() < self.hist["confiabilidad_score"].mean()

    def test_future_probs_sum_to_one(self):
        prob_cols = [f"p_{c}" for c in CLASES]
        sums = self.future[prob_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


class TestPredictEdgeCases:
    def test_cell_not_in_history_no_crash(
        self, pipeline, crime_history_monthly, crime_history_daily, denue, inegi_inv
    ):
        """Celda sin historial → features en 0, sin excepción."""
        unknown = h3.latlng_to_cell(19.0, -104.5, 9)  # lejos de GDL
        result = pipeline.predict(
            HIST_DATE,
            crime_history_monthly,
            crime_history_daily,
            denue,
            inegi_inv,
            h3_indexes=[unknown],
        )
        assert len(result) == 1
        assert result["lambda_diario"].iloc[0] >= 0

    def test_subset_of_cells(
        self, pipeline, h3_cells, crime_history_monthly, crime_history_daily, denue, inegi_inv
    ):
        subset = h3_cells[:5]
        result = pipeline.predict(
            HIST_DATE,
            crime_history_monthly,
            crime_history_daily,
            denue,
            inegi_inv,
            h3_indexes=subset,
        )
        assert len(result) == 5
        assert set(result["h3_index"]) == set(subset)

    def test_empty_daily_history_no_crash(
        self, pipeline, h3_cells, crime_history_monthly, denue, inegi_inv
    ):
        empty_daily = pd.DataFrame(columns=["h3_index", "fecha", "conteo", "categoria"])
        result = pipeline.predict(
            HIST_DATE,
            crime_history_monthly,
            empty_daily,
            denue,
            inegi_inv,
        )
        assert len(result) == len(h3_cells)
        assert (result["lambda_diario"] >= 0).all()

    def test_daily_history_without_categoria(
        self, pipeline, h3_cells, crime_history_monthly, crime_history_daily, denue, inegi_inv
    ):
        """Sin columna categoria en daily history → near-repeat por categoría en 0."""
        daily_no_cat = crime_history_daily.drop(columns=["categoria"])
        result = pipeline.predict(
            HIST_DATE,
            crime_history_monthly,
            daily_no_cat,
            denue,
            inegi_inv,
        )
        assert len(result) == len(h3_cells)


class TestPredictFromFeatures:
    """Path rápido: features precomputadas → E2 → E3."""

    @pytest.fixture(autouse=True)
    def _result(self, pipeline, precomputed_features):
        self.result = pipeline.predict_from_features(HIST_DATE, precomputed_features)

    def test_output_columns(self):
        assert list(self.result.columns) == OUTPUT_COLS

    def test_lambda_non_negative(self):
        assert (self.result["lambda_diario"] >= 0).all()

    def test_probs_sum_to_one(self):
        prob_cols = [f"p_{c}" for c in CLASES]
        sums = self.result[prob_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_confidence_score_range(self):
        assert self.result["confiabilidad_score"].between(0, 100).all()

    def test_categoria_pred_valid(self):
        assert self.result["categoria_pred"].isin(CLASES).all()

    def test_row_count(self, precomputed_features):
        assert len(self.result) == len(precomputed_features)

    def test_future_date_lower_confidence(self, pipeline, precomputed_features):
        # data_end debe pasarse explícitamente en el path precomputado;
        # sin él el pipeline no puede distinguir histórico de futuro.
        data_end = "2023-12-31"
        future = pipeline.predict_from_features(FUTURE_DATE, precomputed_features, data_end=data_end)
        hist   = pipeline.predict_from_features(HIST_DATE,   precomputed_features, data_end=data_end)
        assert future["confiabilidad_score"].mean() < hist["confiabilidad_score"].mean()
