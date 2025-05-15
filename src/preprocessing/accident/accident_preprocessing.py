import pandas as pd
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")

# Load and select relevant columns
accident_df = pd.read_csv(
    f"{config['RAW_DATA_DIR']}/accident.csv", parse_dates=["ACCIDENT_DATE"]
)
accident_df = accident_df[
    [
        "ACCIDENT_NO",
        "ACCIDENT_DATE",
        "ACCIDENT_TIME",
        "DAY_OF_WEEK",
        "ACCIDENT_TYPE",
        "LIGHT_CONDITION",
        "ROAD_GEOMETRY",
        "SPEED_ZONE",
        "SEVERITY",
    ]
].copy()

# Create weekend flag feature
accident_df["WEEKEND"] = (
    accident_df["ACCIDENT_DATE"].dt.weekday.isin([5, 6]).astype(int)
)


# Time of day buckets
def categorize_time_of_day(t):
    h = int(t.split(":")[0])
    if 5 <= h < 12:
        return "MORNING"
    elif 12 <= h < 17:
        return "AFTERNOON"
    elif 17 <= h < 21:
        return "EVENING"
    else:
        return "NIGHT"


# Discretize time and add add time code column
accident_df["TIME_CAT"] = accident_df["ACCIDENT_TIME"].apply(categorize_time_of_day)
accident_df["TIME_CAT_CODE"] = accident_df["TIME_CAT"].map(
    {"NIGHT": 0, "MORNING": 1, "AFTERNOON": 2, "EVENING": 3}
)

# Add IS_PEAK feature
accident_df["IS_PEAK"] = (
    accident_df["TIME_CAT"].isin(["MORNING", "EVENING"]).astype(int)
)

# dDop raw date/time fields
accident_df = accident_df.drop(
    columns=["ACCIDENT_DATE", "ACCIDENT_TIME"], errors="ignore"
)


# Speed numeric (km/h), invalid codes (such as 777/888/999) given as NaN to be handled later
def map_speed(code):
    try:
        s = int(code)
        return s if 40 <= s <= 110 else np.nan
    except:
        return np.nan


accident_df["SPEED_KMH"] = accident_df["SPEED_ZONE"].astype(str).apply(map_speed)
accident_df = accident_df.drop(columns=["SPEED_ZONE"], errors="ignore")

# Drop tiny proportion of severe, potentially misleading data found in the 75km/hr range
accident_df = accident_df[accident_df["SPEED_KMH"] != 75]

# Drop severity level 4 and convert target to binary target "HIGH_SEVERITY"
accident_df = accident_df[accident_df["SEVERITY"] != 4]
accident_df["HIGH_SEVERITY"] = accident_df["SEVERITY"].isin([1, 2]).astype(int)
accident_df = accident_df.drop(columns=["SEVERITY"], errors="ignore")

# One‐hot encode all remaining categoricals
to_ohe = [
    "DAY_OF_WEEK",
    "ACCIDENT_TYPE",
    "LIGHT_CONDITION",
    "ROAD_GEOMETRY",
    "TIME_CAT",
]

accident_df = pd.get_dummies(
    accident_df, columns=to_ohe, prefix_sep="_", drop_first=True
)

# Impute any leftover NaNs (for examnple SPEED_KMH where code was 777/888/999)
accident_df = accident_df.fillna({"SPEED_KMH": accident_df["SPEED_KMH"].median()})
accident_df = accident_df.fillna(0)

# Save out
accident_df.to_csv(
    f"{config['PROCESSED_DATA_DIR']}/preprocessed_accident.csv", index=False
)
