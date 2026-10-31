# Used Car Price Prediction

Machine learning regression project for estimating the market price of used cars based on vehicle characteristics.

## Project Overview

The goal of this project is to develop a machine learning model that predicts the price of a used car in US dollars.

The project follows a complete machine learning workflow:

- exploratory data analysis;
- data cleaning;
- feature engineering;
- preprocessing;
- model training;
- model comparison;
- model evaluation;
- final model selection and serialization.

The original dataset contains **56,244 used cars and 12 columns**.

After cleaning unrealistic extreme values, **55,889 vehicles** remained for model development.

## Dataset

The dataset contains the following information:

- manufacturer;
- model;
- price in USD;
- manufacturing year;
- vehicle condition;
- mileage;
- fuel type;
- engine volume;
- color;
- transmission;
- drive unit;
- vehicle segment.

The target variable is:

`priceUSD`

## Data Cleaning

The cleaning process includes:

- standardizing column names;
- converting numerical columns to the correct data types;
- standardizing categorical text values;
- removing rows with missing essential information;
- removing clearly unrealistic price values;
- removing extreme mileage values;
- removing unrealistic engine sizes.

Only **355 rows** were removed from the original 56,244 records, meaning that more than 99% of the dataset was retained.

Missing values in non-essential columns are handled later through the preprocessing pipeline.

## Feature Engineering

Several additional variables were created to improve the information available to the models.

### Car Age

Vehicle age is calculated relative to the dataset period using 2020 as the reference year.

```python
car_age = 2020 - year
```

### Mileage Per Year

Average annual mileage is calculated using:

```python
mileage_per_year = mileage_km / car_age
```

### Engine Volume in Liters

Engine volume is converted from cubic centimeters to liters.

### Newer Vehicle Indicator

Vehicles manufactured in 2010 or later are marked using a binary feature.

### High Mileage Indicator

Cars with at least 250,000 kilometers are identified using a binary feature.

### Brand and Model

Manufacturer and model information are combined into a new `brand_model` feature.

## Preprocessing

Numerical and categorical variables are processed separately.

Numerical features use:

- median imputation for missing values;
- standard scaling.

Categorical features use:

- most-frequent-value imputation;
- one-hot encoding;
- automatic handling of previously unseen categories.

The entire preprocessing process is implemented using Scikit-learn pipelines.

## Target Transformation

Used-car prices are strongly right-skewed.

To reduce the influence of expensive outliers during training, the target variable is transformed using:

```python
np.log1p(priceUSD)
```

Predictions are automatically transformed back into the original USD price scale using:

```python
np.expm1()
```

## Models Evaluated

Three regression algorithms were evaluated using the same training and testing data.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | $1,098.16 | $2,734.31 | 0.8909 |
| Ridge Regression | $1,254.90 | $3,110.51 | 0.8588 |
| Decision Tree | $1,360.39 | $3,329.84 | 0.8382 |

## Best Model

**Random Forest Regressor** achieved the strongest overall performance.

Final evaluation:

```text
MAE:  $1,098.16
RMSE: $2,734.31
R²:   0.8909
```

An MAE of approximately **$1,100** means that the model's predicted price differs from the actual listed price by about $1,100 on average.

An R² score of approximately **0.89** indicates that the model explains a large proportion of the variation in used-car prices contained in the dataset.

Random Forest was therefore selected as the final model.

## Project Structure

```text
Sasa_Ivanovic_Car_Price_Prediction/
│
├── data/
│   ├── cars.csv
│   ├── cars_clean.csv
│   └── cars_features.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
│
├── models/
│   └── car_price_model.joblib
│
├── model_comparison_results.csv
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project requires:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- jupyter

## Running the Project

Commands should be executed from the root project directory.

### Data Cleaning

```bash
python src/data_cleaning.py
```

This creates:

```text
data/cars_clean.csv
```

### Feature Engineering

```bash
python src/feature_engineering.py
```

This creates:

```text
data/cars_features.csv
```

### Compare Models

```bash
python src/model_comparison.py
```

This trains and evaluates Ridge Regression, Decision Tree and Random Forest models.

The comparison is also saved as:

```text
model_comparison_results.csv
```

### Evaluate the Selected Model

```bash
python src/model_evaluation.py
```

This displays the final Random Forest evaluation metrics and example predictions.

### Train the Final Model

```bash
python src/model_training.py
```

The trained model is saved as:

```text
models/car_price_model.joblib
```

## Jupyter Notebook

The complete exploratory analysis and machine learning workflow is available in:

```text
notebooks/01_eda.ipynb
```

The notebook includes:

- dataset inspection;
- missing-value analysis;
- price distribution;
- data cleaning;
- feature engineering;
- price versus year visualization;
- price versus mileage visualization;
- train/test split;
- training of three regression models;
- model comparison;
- MAE, MSE, RMSE and R² evaluation;
- actual versus predicted price table;
- actual versus predicted price visualization;
- final conclusions.

## Results

The project demonstrates that used-car prices can be estimated with relatively strong accuracy using information such as:

- manufacturer and model;
- manufacturing year;
- mileage;
- engine size;
- fuel type;
- transmission;
- drive unit;
- condition;
- vehicle segment.

The model is not expected to perfectly predict every market price because used-car values can also depend on information that is not available in the dataset, such as maintenance history, accident history, local demand and individual vehicle condition.

## Conclusion

Random Forest produced the best balance of prediction error and explained variance among the evaluated models.

With an MAE of approximately **$1,098** and an R² score of approximately **0.891**, the model provides a useful automatic estimate of used-car market prices.

The project demonstrates a complete machine learning regression workflow, from raw data and exploratory analysis to preprocessing, model comparison, evaluation and saving a reusable trained model.