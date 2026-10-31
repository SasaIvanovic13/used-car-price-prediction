import pandas as pd

from data_cleaning import clean_data


REFERENCE_YEAR = 2020


def add_features(df):
    df = df.copy()

    # Age relative to the dataset period.
    # The newest cars in the dataset are from 2019.
    df["car_age"] = (
        REFERENCE_YEAR - df["year"]
    ).clip(lower=1)

    df["mileage_per_year"] = (
        df["mileage_km"]
        /
        df["car_age"]
    )

    df["engine_volume_liters"] = (
        df["engine_volume_cm3"]
        /
        1000
    )

    df["is_newer_car"] = (
        df["year"] >= 2010
    ).astype(int)

    df["is_high_mileage"] = (
        df["mileage_km"] >= 250000
    ).astype(int)

    df["brand_model"] = (
        df["make"].fillna("unknown").astype(str)
        + "_"
        + df["model"].fillna("unknown").astype(str)
    )

    return df


if __name__ == "__main__":

    data = pd.read_csv(
        "data/cars.csv"
    )

    data = clean_data(data)
    data = add_features(data)

    data.to_csv(
        "data/cars_features.csv",
        index=False
    )

    print(
        "Feature engineered dataset saved."
    )

    print(
        data[
            [
                "make",
                "model",
                "year",
                "car_age",
                "mileage_per_year",
                "engine_volume_liters"
            ]
        ].head()
    )