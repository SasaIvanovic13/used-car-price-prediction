import numpy as np
import pandas as pd

from sklearn.compose import (
    TransformedTargetRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_cleaning import clean_data
from feature_engineering import add_features
from data_preprocessing import (
    FEATURES,
    build_preprocessor
)


data = pd.read_csv(
    "data/cars.csv"
)

data = clean_data(data)
data = add_features(data)


X = data[FEATURES]
y = data["priceUSD"]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)


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


model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)

predictions = np.maximum(
    predictions,
    0
)


mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


print("FINAL MODEL EVALUATION")
print("=" * 60)

print(
    f"MAE:  ${mae:,.2f}"
)

print(
    f"MSE:  {mse:,.2f}"
)

print(
    f"RMSE: ${rmse:,.2f}"
)

print(
    f"R2:    {r2:.4f}"
)


comparison = pd.DataFrame({
    "Actual Price": y_test.iloc[:10].values,
    "Predicted Price": predictions[:10]
})

comparison["Absolute Error"] = (
    comparison["Actual Price"]
    -
    comparison["Predicted Price"]
).abs()


print(
    "\nExample predictions:"
)

print(
    comparison.round(2).to_string(
        index=False
    )
)