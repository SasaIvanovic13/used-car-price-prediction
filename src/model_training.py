import joblib
import numpy as np
import pandas as pd

from sklearn.compose import (
    TransformedTargetRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import (
    FEATURES,
    build_preprocessor
)


DATA_FILE = "data/cars.csv"

MODEL_FILE = (
    "models/car_price_model.joblib"
)


data = pd.read_csv(
    DATA_FILE
)

data = clean_data(data)
data = add_features(data)


X = data[FEATURES]
y = data["priceUSD"]


pipeline = Pipeline([
    (
        "preprocessor",
        build_preprocessor()
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=60,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])


final_model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)


print(
    f"Training final model on "
    f"{len(data)} cars..."
)


final_model.fit(
    X,
    y
)


joblib.dump(
    final_model,
    MODEL_FILE
)


print(
    f"Model saved as "
    f"{MODEL_FILE}"
)