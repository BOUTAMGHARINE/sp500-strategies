# S&P 500 Trading Strategies

## Overview

This project aims to develop and evaluate machine learning-based trading strategies using historical stock market data.

The objective is to predict the direction of future stock returns using technical indicators and a Logistic Regression model. The predictions are then used to generate trading signals and evaluate the performance of the resulting strategy through backtesting.

The project follows a chronological approach to avoid data leakage and preserve the temporal structure of financial data.

## Project Structure

```text
sp500-strategies/
│
├── data/
│   ├── HistoricalData.csv
│   └── all_stocks_5yr.csv
│
├── results/
│   ├── cross-validation/
│   │   ├── metric_train.png
│   │   ├── ml_metrics_train.csv
│   │   └── top_10_feature_importance.csv
│   │
│   ├── selected-model/
│   │   ├── ml_signal.csv
│   │   ├── selected_model.pkl
│   │   └── selected_model.txt
│   │
│   └── strategy/
│       ├── report.md
│       ├── results.csv
│       └── strategy.png
│
├── scripts/
│   ├── features_engineering.py
│   ├── gridsearch.py
│   ├── model_selection.py
│   ├── create_signal.py
│   └── strategy.py
│
├── requirements.txt
└── README.md
```

## Datasets

The project uses two historical financial datasets:

### 1. all_stocks_5yr.csv

Contains historical stock market data for multiple companies, including:

- Date
- Open
- High
- Low
- Close
- Volume
- Name (stock symbol)

This dataset is used for feature engineering and machine learning.

### 2. HistoricalData.csv

Contains historical S&P 500 index data, including:

- Date
- Open
- High
- Low
- Close

This dataset provides historical information about the S&P 500 index.

## Methodology

The project is divided into several stages.

### 1. Data Preprocessing

The raw datasets are loaded and cleaned.

Main operations include:

- Converting date columns into datetime format.
- Handling missing values.
- Sorting stock data chronologically by stock symbol and date.
- Preparing the datasets for feature engineering.

### 2. Feature Engineering

Three technical indicators are calculated for each stock.

| Indicator | Description |
|---|---|
| RSI (14) | Measures the strength and speed of recent price movements. |
| MACD | Measures momentum using the difference between two exponential moving averages. |
| Bollinger Bandwidth | Measures market volatility using Bollinger Bands. |

The future return is also calculated:

\[
Return(D+1,D+2)=\frac{P_{D+2}}{P_{D+1}}-1
\]

The target variable is defined as:

\[
Target_D =
\begin{cases}
1 & \text{if } Return(D+1,D+2)>0\\
0 & \text{otherwise}
\end{cases}
\]

The target indicates whether the future return is positive or non-positive.

**Important:** The future return is used only to construct the target variable. It is never included among the model's input features, in order to prevent data leakage.

### 3. Train/Test Split

The dataset is divided chronologically:

- **Training data:** Dates before January 1, 2017.
- **Test data:** Dates from January 1, 2017 onwards.

The test dataset is kept separate during model selection and hyperparameter tuning.

### 4. Model Selection and Hyperparameter Tuning

A Logistic Regression classifier is used to predict the direction of future returns.

The model is implemented using a Scikit-learn Pipeline containing:

1. StandardScaler: Standardizes the input features.
2. LogisticRegression: Performs binary classification.

GridSearchCV is used to search for the best hyperparameters.

The parameters explored include:

- `C`: Regularization strength.
- `class_weight`: Class weighting strategy.

The model is evaluated using ROC-AUC.

### 5. Time Series Cross-Validation

To respect the chronological structure of financial data, the project uses time-based cross-validation.

The validation strategy includes:

- At least 10 chronological validation folds.
- An initial training period of more than two years.
- Expanding training windows.
- Validation periods occurring strictly after their corresponding training periods.
- All observations from the same date remain in the same fold.

This approach prevents future observations from being used to predict past observations.

### 6. Trading Signal Generation

The selected model is used to generate predictions from the technical indicators.

The predictions are converted into trading signals:

- `1`: Positive signal.
- `0`: Non-positive signal.

The generated signals are saved for subsequent strategy evaluation.

### 7. Backtesting

The trading signals are used to simulate the trading strategy on historical data.

The backtesting stage evaluates the strategy's performance and produces visual and numerical results.

The outputs include:

- Strategy performance visualization.
- Backtesting results in CSV format.
- A performance report.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd sp500-strategies
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution

Run the project scripts in the following logical order:

```bash
python3 scripts/features_engineering.py
python3 scripts/gridsearch.py
python3 scripts/model_selection.py
python3 scripts/create_signal.py
python3 scripts/strategy.py
```

Each stage prepares the outputs required by the subsequent stage.

## Results

The project generates several outputs organized into three categories.

### Cross-validation

- `ml_metrics_train.csv`: Cross-validation evaluation metrics.
- `metric_train.png`: Visualization of model performance.
- `top_10_feature_importance.csv`: Most important features, where supported by the selected model.

### Selected Model

- `selected_model.pkl`: Serialized trained model.
- `selected_model.txt`: Model configuration and information.
- `ml_signal.csv`: Generated trading signals.

### Strategy

- `results.csv`: Backtesting results.
- `strategy.png`: Strategy performance visualization.
- `report.md`: Summary and analysis of the backtesting results.

## Technologies

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Important Considerations

- Financial time-series data must be processed chronologically.
- Future information must never be included in model input features.
- Cross-validation must preserve temporal ordering.
- The test dataset must remain untouched during model selection.
- Model performance should be evaluated using multiple metrics rather than relying exclusively on accuracy or ROC-AUC.

## Author

**Anouar Boutamgharine**