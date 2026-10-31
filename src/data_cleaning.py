import numpy as np
import pandas as pd


def clean_data(df):
    df = df.copy()

    # Standardize column names
    df.columns = [
        column.strip()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        for column in df.columns
    ]

    df = df.rename(
        columns={
            "mileagekilometers": "mileage_km",
            "volumecm3": "engine_volume_cm3"
        }
    )

    # Convert numerical columns
    numeric_columns = [
        "priceUSD",
        "year",
        "mileage_km",
        "engine_volume_cm3"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Standardize categorical values
    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment"
    ]

    for column in categorical_columns:
        df[column] = df[column].apply(
            lambda value:
            value.strip().lower()
            if isinstance(value, str)
            else value
        )

    # Remove rows with missing essential values
    df = df.dropna(
        subset=[
            "priceUSD",
            "year",
            "mileage_km",
            "make",
            "model"
        ]
    ).copy()

    # Remove clearly unrealistic extreme values
    df = df[
        df["priceUSD"].between(
            100,
            150000
        )
    ]

    df = df[
        df["year"].between(
            1950,
            2019
        )
    ]

    df = df[
        df["mileage_km"].between(
            0,
            1_500_000
        )
    ]

    df = df[
        df["engine_volume_cm3"].isna()
        |
        df["engine_volume_cm3"].between(
            500,
            8000
        )
    ]

    return df.reset_index(drop=True)


if __name__ == "__main__":

    input_file = "data/cars.csv"
    output_file = "data/cars_clean.csv"

    data = pd.read_csv(input_file)

    print(
        "Original rows:",
        len(data)
    )

    cleaned_data = clean_data(data)

    print(
        "Rows after cleaning:",
        len(cleaned_data)
    )

    cleaned_data.to_csv(
        output_file,
        index=False
    )

    print(
        f"Clean dataset saved as {output_file}"
    )