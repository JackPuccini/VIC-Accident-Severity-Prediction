import pandas as pd
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")

# Load and drop unused colums
person = pd.read_csv(f"{config['RAW_DATA_DIR']}/person.csv").drop(
    columns=["INJ_LEVEL", "TAKEN_HOSPITAL", "EJECTED_CODE", "LICENCE_STATE"],
    errors="ignore",
)

# AGE ordinal mapping map
AGE_BIN_MAP = {
    "0-4": "CHILD",
    "5-12": "CHILD",
    "13-15": "TEEN",
    "16-17": "TEEN",
    "18-21": "YOUNG",
    "22-25": "YOUNG",
    "26-29": "ADULT",
    "30-39": "ADULT",
    "40-49": "MIDDLE",
    "50-59": "MIDDLE",
    "60-64": "SENIOR",
    "65-69": "SENIOR",
    "70+": "ELDERLY",
    "Unknown": np.nan,
}
AGE_CODE_MAP = {
    "CHILD": 0,
    "TEEN": 1,
    "YOUNG": 2,
    "ADULT": 3,
    "MIDDLE": 4,
    "SENIOR": 5,
    "ELDERLY": 6,
}

person["AGE_CODE"] = person["AGE_GROUP"].map(AGE_BIN_MAP).map(AGE_CODE_MAP)

# UNPROTECTED flag for everyone
PROTECTIVE = {
    1,
    3,
    6,
}  # 1: Seatbelt worn, 3: Child restraint worn, 6: Crash helmet worn
NOT_PROTECTIVE = {
    2,
    4,
    5,
    7,
}  # 2: Seatbelt not worn, 4: Child restraint not worn, 5: Seatbelt/restraint not fitted, 7: Crash helmet not worn

person["UNPROT_FLAG"] = person["HELMET_BELT_WORN"].apply(
    lambda x: 0 if x in PROTECTIVE else 1 if x in NOT_PROTECTIVE else np.nan
)

# Female driver flag
person["IS_FEMALE"] = (person["SEX"] == "F").astype(int)

# Driver only summary
drivers = person[person["ROAD_USER_TYPE_DESC"] == "Drivers"]
driver_summary = drivers.groupby("ACCIDENT_NO", as_index=False).agg(
    YOUNGEST_DRIVER_BIN=("AGE_CODE", "min"),  # Youngest age group involved
    UNPROT_FRAC_DRIVER=("UNPROT_FLAG", "mean"),  # Proportion of unprotected drivers
    DRIVER_FEMALE_PCT=(
        "IS_FEMALE",
        "mean",
    ),  # Gender proportions involved (lower values means more males)
)

# All person summary
all_summary = person.groupby("ACCIDENT_NO", as_index=False).agg(
    N_PERSONS=("UNPROT_FLAG", "count"),
    N_UNPROTECTED=("UNPROT_FLAG", lambda col: col.eq(1).sum()),
)
all_summary["UNPROT_ALL_FRAC"] = all_summary["N_UNPROTECTED"] / all_summary["N_PERSONS"]

# Merge driver and all summaries
person_summary = driver_summary.merge(all_summary, on="ACCIDENT_NO", how="outer")

# Fill in accidents with no drivers
person_summary[
    ["YOUNGEST_DRIVER_BIN", "UNPROT_FRAC_DRIVER", "DRIVER_FEMALE_PCT"]
] = person_summary[
    ["YOUNGEST_DRIVER_BIN", "UNPROT_FRAC_DRIVER", "DRIVER_FEMALE_PCT"]
].fillna(
    {
        "YOUNGEST_DRIVER_BIN": -1,  # Filler value to avoid NaNs
        "UNPROT_FRAC_DRIVER": 0.0,
        "DRIVER_FEMALE_PCT": 0.0,
    }
)

# Save
person_summary.to_csv(
    f"{config['PROCESSED_DATA_DIR']}/preprocessed_person.csv", index=False
)
