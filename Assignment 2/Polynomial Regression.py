import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Feature selection from scratch mesh bult in using correlation
def Feature_Selector(data, Label, threshold=0.5):
    correlations = data.corr()[Label].abs()
    correlations = correlations.drop(Label)
    return correlations[correlations >= threshold].index.tolist()

# Feature scaling from scratch mesh bult in using normalization
def feature_scaling(X_train, X_test):
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds
    return X_train_scaled, X_test_scaled

# Making polynomial features using a recursive function to calculate all possible combinations for the degree entered by the user
def poly_Features(X, degree):
    Nofsamples, Noffeatures = X.shape
    features = [np.ones(Nofsamples)]
    names = ['1']
    
    def Recurse(remaining_degree, start, term, name):
        if remaining_degree < 1: return
        for i in range(start, Noffeatures):
            feature = X[:, i]
            for exp in range(1, remaining_degree + 1):
                new_term = term * (feature ** exp)
                new_name = f"{name} × " if name else ""
                new_name += f"x{i}" + (f"^{exp}" if exp > 1 else "")
                features.append(new_term)
                names.append(new_name)
                Recurse(remaining_degree - exp, i + 1, new_term.copy(), new_name)
    
    Recurse(degree, 0, np.ones(Nofsamples), "")
    return np.column_stack(features), names

# Load data
data = pd.read_csv('assignment1dataset.csv').dropna()

# Feature Selection using my function
train_data = data.iloc[:, 1:].join(data['RevenuePerDay'])
selected_features = Feature_Selector(train_data, 'RevenuePerDay', threshold=0.5)
print("Selected Features:", selected_features)
data_filtered = data[selected_features + ['RevenuePerDay']]

# Feature Encoding for categorical variables
X = data_filtered.iloc[:, :-1]
Y = data_filtered['RevenuePerDay']
categorical = X.select_dtypes(include=['object', 'category']).columns

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)

# Feature Scaling using my function
X_train_scaled, X_test_scaled = feature_scaling(X_train.values, X_test.values)

# Normalize MSE 
y_variance = np.var(Y)

# Polynomial regression 
degrees = range(1, 17)  
train_results = []
test_results = []

for degree in degrees:
    X_poly_train, _ = poly_Features(X_train_scaled, degree)
    X_poly_test, _ = poly_Features(X_test_scaled, degree)
    
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_poly_train, Y_train)
    
    train_mse = metrics.mean_squared_error(Y_train, model.predict(X_poly_train))
    test_mse = metrics.mean_squared_error(Y_test, model.predict(X_poly_test))
    
    train_mse_normalized = train_mse / y_variance
    test_mse_normalized = test_mse / y_variance
    
    train_results.append(train_mse_normalized)
    test_results.append(test_mse_normalized)
    
    print(f"Degree {degree}: Train MSE = {train_mse_normalized:.4f}, Test MSE = {test_mse_normalized:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_results, label='Train MSE', marker='o')
plt.plot(degrees, test_results, label='Test MSE', marker='x')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Polynomial Degree vs MSE (Train and Test)')
plt.legend()
plt.grid(True)
plt.show()