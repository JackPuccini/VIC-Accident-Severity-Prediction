import pandas as pd
from dotenv import dotenv_values

config = dotenv_values(".env")

# 1) Load
veh = pd.read_csv(f"{config['RAW_DATA_DIR']}/vehicle.csv")

# 2) Drop unwanted cols
veh.drop(
    columns=[
        "VEHICLE_ID",
        "LEVEL_OF_DAMAGE",
        "TOWED_AWAY_FLAG",
        "CAUGHT_FIRE",
        "ROAD_SURFACE_TYPE_DESC",
        "TRAFFIC_CONTROL_DESC",
        "VEHICLE_MAKE",
        "VEHICLE_MODEL",
        "VEHICLE_COLOUR_2",
        "REG_STATE",
        "VEHICLE_POWER",
        "VEHICLE_WEIGHT",
        "VEHICLE_YEAR_MANUF",  # Can keep this later when process properly
    ],
    errors="ignore",
    inplace=True,
)

# 3) Toss the  'G' = “unknown what is being towed” rows:
veh = veh[veh["TRAILER_TYPE"] != "G"]

# 4) Single boolean flag: H means “no trailer”, everything else (A–L, I, J, K, etc.) → tow
veh["HAS_TRAILER"] = (veh["TRAILER_TYPE"] != "H").astype(int)

# 5) Vehicle count for aggregation
veh["VEHICLE_COUNT"] = 1

# 6) Per-accident aggregation:
#    - HAS_TRAILER → max (0 if none, 1 if any)
#    - VEHICLE_COUNT → sum
veh_sum = veh.groupby("ACCIDENT_NO", as_index=False).agg(
    ANY_TRAILER=("HAS_TRAILER", "max"),
    NUM_VEHICLES=("VEHICLE_COUNT", "sum"),
)

# 7) Save
veh_sum.to_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_vehicle.csv", index=False)
