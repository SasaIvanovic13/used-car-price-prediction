import numpy as np
import pandas as pd

from sklearn.compose import (
    TransformedTargetRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
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


def create_model(regressor):

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "regressor",
            regressor
        )
    ])

    # Price distribution is strongly skewed,
    # so the target is trained in logarithmic form.
    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    return model


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


models = {
    "Ridge Regression": Ridge(
        alpha=10.0
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=25,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=60,
        max_depth=24,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}


results = []


for name, regressor in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model = create_model(
        regressor
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

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"MAE:  ${mae:,.2f}"
    )

    print(
        f"RMSE: ${rmse:,.2f}"
    )

    print(
        f"R2:    {r2:.4f}"
    )


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="MAE"
).reset_index(drop=True)


print("\nMODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)