import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    print("Error: scikit-learn is not installed. Please install it using: pip install scikit-learn")
    raise

import os
if not os.path.exists('house_prices.csv'):
    print("house_prices.csv not found. Generating sample data...")
    np.random.seed(42)
    n_samples = 100
    area = np.random.randint(1000, 4000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    # Simple price model: price = 100 * area + 20000 * bedrooms + 15000 * bathrooms + noise
    price = 100 * area + 20000 * bedrooms + 15000 * bathrooms + np.random.normal(0, 50000, n_samples)
    sample_data = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'price': price})
    sample_data.to_csv('house_prices.csv', index=False)
    print("Sample data generated and saved to house_prices.csv")

data = pd.read_csv('house_prices.csv')

print("Missing values in each column:")
print(data.isnull().sum())


data = data.fillna(data.mean())

print("\nSummary statistics:")
print(data.describe())

correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Correlation matrix plot saved as correlation_matrix.png")
plt.close()

# Scatter plots for key features vs price
features = ['area', 'bedrooms', 'bathrooms']  # Adjust based on your data
for feature in features:
    plt.figure()
    plt.scatter(data[feature], data['price'])
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title(f'{feature} vs Price')
    plt.savefig(f'{feature}_vs_price.png')
    print(f"Plot saved as {feature}_vs_price.png")
    plt.close()

# Step 4: Feature Selection
# Select features with high correlation to price (absolute value > 0.5)
target_corr = correlation_matrix['price'].abs()
selected_features = target_corr[target_corr > 0.5].index.tolist()
selected_features.remove('price')  # Remove target variable
print(f"\nSelected features: {selected_features}")

X = data[selected_features]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

print("\nModel Coefficients:")
for feature, coef in zip(selected_features, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

new_house_data = {}
for feature in selected_features:
    if feature == 'area':
        new_house_data[feature] = 2000
    elif feature == 'bedrooms':
        new_house_data[feature] = 3
    elif feature == 'bathrooms':
        new_house_data[feature] = 2

new_house = pd.DataFrame([new_house_data])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for new house: ${predicted_price[0]:.2f}")