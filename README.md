# California Housing Price Prediction
 
A machine learning mini-project that predicts California housing prices using Linear Regression and Random Forest in Python, built on the classic `scikit-learn` California Housing dataset.
 
---
 
## Project Structure
 
```
├── task1_ml_linear_regression.ipynb   # Main notebook (EDA, modeling, evaluation)
└── Linear_Regression_Report.pdf       # Written summary of findings
```
 
---
 
## Workflow
 
The notebook walks through a complete regression pipeline:
 
1. **Data Loading & Inspection** — shape, columns, null checks, and descriptive statistics
2. **Exploratory Data Analysis** — distribution plots and feature correlation heatmap
3. **Train/Test Split** — standard 80/20 partition
4. **Model Training** — `LinearRegression` from scikit-learn
5. **Evaluation** — MAE, RMSE, and R² score
6. **Feature Scaling** — re-evaluating the model after scaling (results are identical for OLS, as expected)
7. **Model Export** — saving the trained model via `pickle`
8. **Random Forest Comparison** — initial `RandomForestRegressor` setup for benchmarking
---
 
## Results
 
| Metric | Score |
|--------|-------|
| MAE | 0.5332 |
| RMSE | 0.7456 |
| R² | 0.5758 |
 
> Scaled and unscaled linear regression produce identical metrics — expected behavior for ordinary least squares.
 
---
 
## Tech Stack
 
| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Modeling & evaluation |
| `pickle` | Model serialization |
| Jupyter Notebook | Interactive development |
 
---
 
## Getting Started
 
**Install dependencies:**
 
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```
 
**Run the notebook:**
 
```bash
jupyter notebook task1_ml_linear_regression.ipynb
```
 
Run all cells in order. A `linear_regression_model.pkl` file will be saved locally once the model export cell executes.
 
---
 
## Notes
 
- A `pip install seaborn` cell is included at the top of the notebook for convenience.
- The Random Forest section is present in the notebook but its evaluation output was not captured in the saved state — re-run the notebook to generate it.
 
