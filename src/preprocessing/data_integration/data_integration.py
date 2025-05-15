import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import dotenv_values

config = dotenv_values(".env")

# -------------------- Data Integration ----------------------------

# Load each csv file
veh = pd.read_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_vehicle.csv")
acc = pd.read_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_accident.csv")
prs = pd.read_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_person.csv")
rnd = pd.read_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_road_surface.csv")
atm = pd.read_csv(f"{config['PROCESSED_DATA_DIR']}/preprocessed_atmospheric.csv")

# Merge all dataframes on ACCIDENT_NO, using inner merge for 'essential' data and left merge on road_cond and atmosphere data
df = (
    acc.merge(veh, on="ACCIDENT_NO", how="inner")
    .merge(prs, on="ACCIDENT_NO", how="inner")
    .merge(rnd, on="ACCIDENT_NO", how="left")
    .merge(atm, on="ACCIDENT_NO", how="left")
)

# Fill any remaining holes with zero
df = df.fillna(0)

# Save
df.to_csv(f"{config['PROCESSED_DATA_DIR']}/integrated_data.csv", index=False)

# -------------------- Test - Train Splitting ----------------------------

# Split data into training and testing sets
df = pd.read_csv(f"{config['PROCESSED_DATA_DIR']}/integrated_data.csv")

# Split data, stratifying on HIGH_SEVERITY target variable
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["HIGH_SEVERITY"])

# Save
train_df.to_csv(f"{config['TRAIN_TEST_DATA_DIR']}/train.csv", index=False)
test_df.to_csv(f"{config['TRAIN_TEST_DATA_DIR']}/test.csv", index=False)
