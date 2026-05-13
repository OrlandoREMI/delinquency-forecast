import zipfile
import io
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
INPUT_ZIP = ROOT / "data/inegi_cpv2020/RESAGEBURB_14_2020_csv.zip"
INPUT_CSV = "RESAGEBURB_14CSV20.csv"
OUTPUT = ROOT / "data/processed/manzanas_cpv.parquet"

COLS_SOURCE = [
    "ENTIDAD", "MUN", "LOC", "AGEB", "MZA",
    "POBTOT", "VIVTOT",
    "P_18A24", "POB15_64",
    "PDESOCUP", "PEA",
    "GRAPROES", "P15YM_SE", "P15YM_AN",
    "PNACOE", "PRESOE15",
    "PSINDER",
    "PRO_OCUP_C",
    "VPH_PISOTI", "VPH_NODREN", "VPH_S_ELEC", "VPH_NDEAED", "VPH_SNBIEN", "VPH_1CUART",
    "HOGJEF_F", "TOTHOG",
    "VPH_INTER", "VPH_AUTOM",
    "PROM_HNV",
]

RENAME = {
    "ENTIDAD":    "cve_ent",
    "MUN":        "cve_mun",
    "LOC":        "cve_loc",
    "AGEB":       "cve_ageb",
    "MZA":        "cve_mza",
    "POBTOT":     "pobtot",
    "VIVTOT":     "vivtot",
    "P_18A24":    "p_18a24",
    "POB15_64":   "pob15_64",
    "PDESOCUP":   "pdesocup",
    "PEA":        "pea",
    "GRAPROES":   "graproes",
    "P15YM_SE":   "p15ym_sin_escolaridad",
    "P15YM_AN":   "p15ym_analfabeta",
    "PNACOE":     "pnacoe",
    "PRESOE15":   "presoe15",
    "PSINDER":    "sin_derechohabiencia",
    "PRO_OCUP_C": "ocupantes_por_cuarto",
    "VPH_PISOTI": "viv_piso_tierra",
    "VPH_NODREN": "viv_sin_drenaje",
    "VPH_S_ELEC": "viv_sin_electricidad",
    "VPH_NDEAED": "viv_sin_servicios_basicos",
    "VPH_SNBIEN": "viv_sin_bienes",
    "VPH_1CUART": "viv_un_cuarto",
    "HOGJEF_F":   "hogares_jefa_mujer",
    "TOTHOG":     "tothog",
    "VPH_INTER":  "viv_con_internet",
    "VPH_AUTOM":  "viv_con_auto",
    "PROM_HNV":   "prom_hijos_nacidos_vivos",
}

NUMERIC_COLS = [
    "pobtot", "vivtot", "p_18a24", "pob15_64", "pdesocup", "pea",
    "p15ym_sin_escolaridad", "p15ym_analfabeta", "pnacoe", "presoe15",
    "sin_derechohabiencia", "viv_piso_tierra", "viv_sin_drenaje",
    "viv_sin_electricidad", "viv_sin_servicios_basicos", "viv_sin_bienes",
    "viv_un_cuarto", "hogares_jefa_mujer", "tothog", "viv_con_internet",
    "viv_con_auto",
]

FLOAT_COLS = ["graproes", "ocupantes_por_cuarto", "prom_hijos_nacidos_vivos"]


def build():
    print("Leyendo RESAGEBURB...")
    with zipfile.ZipFile(INPUT_ZIP) as z:
        with z.open(INPUT_CSV) as f:
            df = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                usecols=COLS_SOURCE,
                dtype=str,
            )

    print(f"  Total filas leídas: {len(df):,}")

    df = df[df["MZA"] != "000"].copy()
    print(f"  Filas nivel manzana: {len(df):,}")

    df["cvegeo"] = (
        df["ENTIDAD"].str.zfill(2)
        + df["MUN"].str.zfill(3)
        + df["LOC"].str.zfill(4)
        + df["AGEB"].str.zfill(4)
        + df["MZA"].str.zfill(3)
    )

    df = df.rename(columns=RENAME)

    for col in NUMERIC_COLS + FLOAT_COLS:
        df[col] = df[col].replace("*", pd.NA)

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    key_cols = ["cvegeo", "cve_ent", "cve_mun", "cve_loc", "cve_ageb", "cve_mza"]
    feature_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + feature_cols]

    print("\nValores nulos por columna (asteriscos de confidencialidad):")
    nulls = df[feature_cols].isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  Ninguno")
    else:
        for col, n in nulls.items():
            print(f"  {col}: {n:,} ({n/len(df)*100:.1f}%)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"\nGuardado en {OUTPUT}")
    print(f"Shape final: {df.shape}")


if __name__ == "__main__":
    build()
