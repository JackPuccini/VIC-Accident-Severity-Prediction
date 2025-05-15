import pandas as pd
from dotenv import dotenv_values

config = dotenv_values(".env")

# Load vehicle dataset
veh_df = pd.read_csv(f"{config['RAW_DATA_DIR']}/vehicle.csv")

# Drop unwanted columns
veh_df.drop(
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
        "VEHICLE_YEAR_MANUF",
    ],
    errors="ignore",
    inplace=True,
)

# Drop when value is 'G' = "unknown what is being towed" rows:
veh_df = veh_df[veh_df["TRAILER_TYPE"] != "G"]

# Boolean flag for trailer usage. H means "no trailer", everything else counts as some form of trailer
veh_df["HAS_TRAILER"] = (veh_df["TRAILER_TYPE"] != "H").astype(int)

# Vehicle count for aggregation
veh_df["VEHICLE_COUNT"] = 1

# Per-accident aggregation
veh_sum = veh_df.groupby("ACCIDENT_NO", as_index=False).agg(
    ANY_TRAILER=(
        "HAS_TRAILER",
        "max",
    ),  # Set to '1'/True if any vehicle involved has a trailer
    NUM_VEHICLES=("VEHICLE_COUNT", "sum"),  # Count the number of vehicles involved
)

veh_sum.to_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_vehicle.csv", index=False)
