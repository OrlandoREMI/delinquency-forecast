import pandas as pd
import holidays as hol_lib

_MX_HOLIDAYS: set | None = None


def _get_holidays() -> set:
    global _MX_HOLIDAYS
    if _MX_HOLIDAYS is None:
        _MX_HOLIDAYS = set(hol_lib.Mexico(years=list(range(2017, 2035))).keys())
    return _MX_HOLIDAYS


def date_features(fecha: pd.Timestamp) -> dict:
    return {
        "dia_semana":    fecha.dayofweek,
        "es_fin_semana": int(fecha.dayofweek >= 5),
        "es_festivo":    int(fecha.date() in _get_holidays()),
        "mes":           fecha.month,
    }
