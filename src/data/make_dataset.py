import pandas as pd
from glob import glob


def read_data_from_files(files, data_path):
    accel_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    accel_set = 1
    gyro_set = 1

    # Extract features from filename
    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = accel_set
            accel_set += 1
            accel_df = pd.concat([accel_df, df], ignore_index=True)

        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df], ignore_index=True)

    # Convert timestamps to datetimes
    accel_df.index = pd.to_datetime(accel_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")
    accel_df.drop(columns=["epoch (ms)", "time (01:00)"], inplace=True)
    gyro_df.drop(columns=["epoch (ms)", "time (01:00)"], inplace=True)

    return accel_df, gyro_df


# Merging datasets
files = glob("../../data/raw/MetaMotion/*.csv")
data_path = "../../data/raw/MetaMotion/"
accel_df, gyro_df = read_data_from_files(files, data_path)

accel_df.drop(columns=["elapsed (s)"], inplace=True)
gyro_df.drop(columns=["elapsed (s)"], inplace=True)

merged_data = pd.merge(
    accel_df.reset_index(),
    gyro_df.reset_index(),
    how="outer",
    on=["epoch (ms)", "participant", "label", "category", "set"],
)

merged_data.set_index("epoch (ms)", drop=True, inplace=True)


# Rename columns
merged_data.rename(
    columns={
        "x-axis (g)": "accel_x (g)",
        "y-axis (g)": "accel_y (g)",
        "z-axis (g)": "accel_z (g)",
        "x-axis (deg/s)": "gyro_x (deg/s)",
        "y-axis (deg/s)": "gyro_y (deg/s)",
        "z-axis (deg/s)": "gyro_z (deg/s)",
    },
    inplace=True,
)
merged_data.sort_index(inplace=True)


# Resample data
# Accelerometer: 12.500HZ
# Gyroscope: 25.000Hz
sampling_config = {
    "accel_x (g)": "mean",
    "accel_y (g)": "mean",
    "accel_z (g)": "mean",
    "gyro_x (deg/s)": "mean",
    "gyro_y (deg/s)": "mean",
    "gyro_z (deg/s)": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}


# Split data by days to avoid sample generation in gaps
days = [group for day, group in merged_data.groupby(pd.Grouper(freq="D"))]
resampled_data = pd.concat(
    [df.resample("200ms").apply(sampling_config).dropna() for df in days]
)

resampled_data["set"] = resampled_data["set"].astype("int")


# Export dataset as pickled object so that all the type conversions remain the same
resampled_data.to_pickle("../../data/intermediate/01_processed_data.pkl")
