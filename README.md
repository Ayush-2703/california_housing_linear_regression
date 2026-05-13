# California Housing Linear Regression

This repository contains a machine learning mini-project focused on predicting California housing prices using linear regression in Python.

## Contents

- `task1_ml_linear_regression.ipynb`: Jupyter notebook with data loading, exploratory data analysis, model training, evaluation, feature scaling, and a Random Forest extension.
- `Linear_Regression_Report.pdf`: Project report summarizing the workflow and findings.

## Project Overview

The notebook uses the California Housing dataset from `scikit-learn` and walks through a standard regression workflow:

- Load and inspect the dataset
- Check shape, columns, null values, and descriptive statistics
- Visualize distributions and feature correlations
- Split the data into training and testing sets
- Train a `LinearRegression` model
- Evaluate performance using MAE, RMSE, and R2 score
- Compare results after feature scaling
- Save the trained model with `pickle`
- Begin a comparison with `RandomForestRegressor`

## Reported Linear Regression Results

From the notebook output:

- MAE: `0.5332`
- RMSE: `0.7456`
- R2 Score: `0.5758`

The scaled linear regression model reports the same metrics, which is expected for ordinary least squares regression in this setup.

## Tech Stack

- Python
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Install the required Python libraries.
2. Open `task1_ml_linear_regression.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the notebook cells in order.

A quick install command is:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Notes

- The notebook includes a `pip install seaborn` cell at the top.
- A model artifact may be generated locally as `linear_regression_model.pkl` when the notebook is executed.
- The Random Forest section is present in the notebook, but its final evaluation output is not captured in the saved notebook output.

## License

This project is licensed under the MIT License.
