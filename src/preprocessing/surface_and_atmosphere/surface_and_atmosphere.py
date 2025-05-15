import pandas as pd
from dotenv import dotenv_values

config = dotenv_values(".env")

# --- Road Surface Condition ---
road_df = pd.read_csv(f"{config['RAW_DATA_DIR']}/road_surface_cond.csv")

# 1) Drop the text and sequence columns
road_df = road_df.drop(
    columns=["SURFACE_COND_DESC", "SURFACE_COND_SEQ"], errors="ignore"
)

# 2) Slippery surface flag: codes 2=Wet, 3=Muddy, 4=Snowy, 5=Icy
road_df["SLIPPERY_SURFACE"] = road_df["SURFACE_COND"].isin([2, 3, 4, 5]).astype(int)

# 3) (Optional) One-hot encode the full SURFACE_COND categories
surface_ohe = pd.get_dummies(
    road_df["SURFACE_COND"],
    prefix="SURFACE",
    prefix_sep="_",
    drop_first=False,
    dtype=int,
)
road_df = pd.concat([road_df, surface_ohe], axis=1)

# 5) Save back out
road_df.to_csv(
    f"{config['PROCESSED_DATA_DIR']}/road_surface_preprocessed.csv", index=False
)

# --- Atmospheric Condition ---
atm_df = pd.read_csv(f"{config['RAW_DATA_DIR']}/atmospheric_cond.csv")

# 1) Drop the text and sequence columns
atm_df = atm_df.drop(columns=["ATMOSPH_COND_DESC", "ATMOSPH_COND_SEQ"], errors="ignore")

# 2) Adverse weather flag: codes 2=Raining,3=Snowing,4=Fog,5=Smoke,6=Dust,7=Strong winds
atm_df["ADVERSE_WEATHER"] = atm_df["ATMOSPH_COND"].isin([2, 3, 4, 5, 6, 7]).astype(int)

# 3) (Optional) One-hot encode the ATMOSPH_COND categories
weather_ohe = pd.get_dummies(
    atm_df["ATMOSPH_COND"],
    prefix="WEATHER",
    prefix_sep="_",
    drop_first=False,
    dtype=int,
)

atm_df = pd.concat([atm_df, weather_ohe], axis=1)
atm_df.to_csv(
    f"{config['PROCESSED_DATA_DIR']}/atmospheric_core_preprocessed.csv", index=False
)
