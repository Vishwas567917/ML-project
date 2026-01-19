# House Price Prediction Using Machine Learning

This project demonstrates a complete machine learning workflow for predicting house prices using linear regression in Python.

## Prerequisites

- Python 3.8 or higher
- Install required packages: `pip install -r requirements.txt`

## Dataset

Place your house prices dataset in a CSV file named `house_prices.csv` in the project root. The dataset should include columns such as:
- `area`: Square footage of the house
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `price`: House price (target variable)

Example dataset structure:
```
area,bedrooms,bathrooms,price
2000,3,2,300000
1500,2,1,250000
...
```

## Usage

1. Ensure your dataset is named `house_prices.csv` and placed in the project directory.
2. Run the script: `python house_price_prediction.py`
3. The script will:
   - Load and clean the data
   - Perform exploratory data analysis (plots will be displayed)
   - Select relevant features
   - Train a linear regression model
   - Evaluate the model
   - Provide an example prediction

## Output

- Correlation matrix heatmap
- Scatter plots for features vs price
- Model evaluation metrics (MSE, R-squared)
- Model coefficients
- Example prediction for a new house

## Troubleshooting

- If you encounter import errors, ensure all packages are installed.
- If plots don't display, ensure matplotlib backend is configured (e.g., in Jupyter or with plt.show()).
- Adjust feature selection threshold or add more features as needed for your dataset.