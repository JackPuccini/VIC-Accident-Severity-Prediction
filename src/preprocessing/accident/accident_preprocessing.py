import pandas as pd
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")

# 1) Load & select relevant columns
acc = pd.read_csv(
    f"{config['RAW_DATA_DIR']}/accident.csv", parse_dates=["ACCIDENT_DATE"]
)
acc = acc[
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

# 2) Weekend flag
acc["WEEKEND"] = acc["ACCIDENT_DATE"].dt.weekday.isin([5, 6]).astype(int)


# 3) Time‐of‐day buckets
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


acc["TIME_CAT"] = acc["ACCIDENT_TIME"].apply(categorize_time_of_day)
acc["IS_PEAK"] = acc["TIME_CAT"].isin(["MORNING", "EVENING"]).astype(int)
acc["TIME_CAT_CODE"] = acc["TIME_CAT"].map(
    {"NIGHT": 0, "MORNING": 1, "AFTERNOON": 2, "EVENING": 3}
)

# drop raw date/time fields
acc = acc.drop(columns=["ACCIDENT_DATE", "ACCIDENT_TIME"], errors="ignore")


# 4) Speed numeric (km/h), invalid codes → NaN
def map_speed(code):
    try:
        s = int(code)
        return s if 40 <= s <= 110 else np.nan
    except:
        return np.nan


acc["SPEED_KMH"] = acc["SPEED_ZONE"].astype(str).apply(map_speed)
acc = acc.drop(columns=["SPEED_ZONE"], errors="ignore")

# Drop tiny proportion of severe, potentially misleading data
acc = acc[acc["SPEED_KMH"] != 75]


# 5) Binary target: high‐severity (1=fatal/serious, 0=other)
acc = acc[acc["SEVERITY"] != 4]
acc["HIGH_SEVERITY"] = acc["SEVERITY"].isin([1, 2]).astype(int)
acc = acc.drop(columns=["SEVERITY"], errors="ignore")

# 6) One‐hot encode all remaining categoricals
to_ohe = [
    "DAY_OF_WEEK",
    "ACCIDENT_TYPE",
    "LIGHT_CONDITION",
    "ROAD_GEOMETRY",
    "TIME_CAT",
]
acc = pd.get_dummies(acc, columns=to_ohe, prefix_sep="_", drop_first=True)

# 7) Impute any stray NaNs (e.g. SPEED_KMH where code was 777/888/999)
acc = acc.fillna({"SPEED_KMH": acc["SPEED_KMH"].median()})
acc = acc.fillna(0)

# 8) Save out
acc.to_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_accident.csv", index=False)
