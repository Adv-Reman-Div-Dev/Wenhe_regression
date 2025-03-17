# This works with predicted values and RMSE that are different for each prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from itertools import product
import matplotlib.pyplot as plt

# Delete the Keras Tuner cache directory to ensure a fresh search for hyperparameters
import shutil
shutil.rmtree('keras_tuner', ignore_errors=True)

# Read the CSV file
data = pd.read_csv('D:\\Feng_Wenhe\\Python\\Nano_Honeywell_DoE.csv')

# Assume the CSV file contains 4 input parameters (A-D) and 2 output values (E, F)
X = data[['A', 'B', 'C']].values
y = data[['D']].values  # Only using column 'F' as the target output

# 1. Compute the Pearson correlation matrix to analyze linear relationships between input parameters and output F
correlation_matrix = data[['A', 'B', 'C', 'D']].corr()

# Display the correlation between each input feature and the target variable F
print(correlation_matrix['D'])

# 2. Normalize the data
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Compute the range of the original y values (used for calculating relative RMSE)
y_range = y.max() - y.min()

# 3. ANN
# 3.1 Define a function to build the neural network model (for Keras Tuner optimization)
def build_model(hp):
    model = Sequential()
    # Input layer with a tunable number of neurons
    model.add(Dense(units=hp.Int('input_units', min_value=32, max_value=128, step=32), 
                    input_dim=3, activation='relu'))
    
    # Hidden layers (the number of layers and neurons per layer are optimized by Keras Tuner)
    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=128, step=32), 
                        activation='relu'))
    
    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 3.2 Initialize Keras Tuner for hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Try 10 different model configurations
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='ann_regression'
)


# 3.3 Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()
rmse_scores = []
rrmse_scores = []

# Initialize lists to store predicted and actual values
y_pred_list = []
y_true_list = []

# Perform LOOCV
for train_index, test_index in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    
    # Use Keras Tuner to search for the best model
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Train the best model
    best_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    
    # Make predictions
    y_pred_scaled = best_model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    y_pred_value = scaler_y.inverse_transform(y_pred_scaled)  # Convert back to original scale
    y_true_value = scaler_y.inverse_transform(y_test)  # Convert actual values back to original scale
    
    # Store the predicted and actual values
    y_pred_list.append(y_pred_value[0][0])  # Append predicted scalar value
    y_true_list.append(y_true_value[0][0])  # Append actual scalar value
    
    # Compute RMSE for this fold
    rmse = np.sqrt(mean_squared_error(y_true_value, y_pred_value))
    rmse_scores.append(rmse)
    
    # Compute relative RMSE (rRMSE)
    y_range = np.max(y) - np.min(y)  # Use the range of actual y values for relative RMSE
    rrmse = rmse / y_range
    rrmse_scores.append(rrmse)

    # Retrieve the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Display the optimized hyperparameters
    (f"Optimal number of input units: {best_hp.get('input_units')}")
    for i in range(best_hp.get('n_layers')):
        print(f"Optimal number of units in hidden layer {i + 1}: {best_hp.get(f'layer_{i}_units')}")

# 3.4 Evaluation
# Output the average RMSE and relative RMSE
print(f'LOOCV RMS (ANN regression): {np.mean(rmse_scores)}')
print(f'LOOCV Relative RMSE (ANN regression): {np.mean(rrmse_scores)}')

# Calculate R² score
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_true_list, y_pred_list)
print(f'LOOCV R² (ANN regression): {r2:.4f}')


# 4. Plot of the LOOCV result
# Plot actual values vs. predicted values
sorted_indices = np.argsort(y_true_list)  # Get indices that sort y_true_list

# Sort both actual and predicted values based on these indices
sorted_actual = np.array(y_true_list)[sorted_indices]
sorted_predicted = np.array(y_pred_list)[sorted_indices]

# Create a new figure for the plot
plt.figure(figsize=(8, 6))

# Scatter plot of sorted actual F values
plt.scatter(range(len(sorted_actual)), sorted_actual, color='blue', alpha=0.6, label='Actual D')

# Scatter plot of sorted LOOCV predicted F values
plt.scatter(range(len(sorted_predicted)), sorted_predicted, color='orange', alpha=0.6, label='LOOCV Predicted D')

# Add labels and title
plt.xlabel('Data Points (Sorted)')
plt.ylabel('D Values')
plt.title('Sorted Actual vs. Sorted LOOCV Predicted Depth Values')

# Display grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

# 5. Grid search to generate a database
# Extract unique values of parameters A-D from the CSV file for grid search
param_grid = {
    'A': data['A'].unique(),
    'B': data['B'].unique(),
    'C': data['C'].unique(),
}

# Perform Grid Search to generate all possible combinations of A-D
def grid_search_and_predict(param_grid, model, X_scaled, y_scaled, scaler_X, scaler_y, loo_rmse):
    results = []
    
    # Compute predictions and error ranges for each combination of A-D values
    for params in product(*param_grid.values()):  # Generate all parameter combinations
        # Create a dictionary of current parameter combination (e.g., A=1, B=2, C=3, D=4)
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Retrieve the corresponding X values (A-D)
        X_input = np.array([param_dict[key] for key in param_grid.keys()]).reshape(1, -1)
        X_input_scaled = scaler_X.transform(X_input)  # Normalize the input
        
        # Predict the F value using the trained model
        prediction_scaled = model.predict(X_input_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)  # Convert back to original scale
        
        # Find the closest training data point to this parameter combination
        distances = np.linalg.norm(X_scaled - X_input_scaled, axis=1)
        closest_index = np.argmin(distances)
        
        # Use the LOOCV RMSE of the closest training data point as the error range
        error_range = loo_rmse[closest_index]
        
        # Compute relative RMSE (rRMSE)
        y_range = np.max(y) - np.min(y)  # Compute range of actual y values
        rrmse = error_range / y_range
        
        # Store the results
        results.append({
            **param_dict, 
            'Predicted_D': prediction[0][0], 
            'RMSE': error_range,  
            'rRMSE': rrmse  
        })
    
    # Return results as a DataFrame for better visualization
    return pd.DataFrame(results)

# Perform Grid Search and prediction
results_df = grid_search_and_predict(param_grid, best_model, X_scaled, y_scaled, scaler_X, scaler_y, rmse_scores)

# Save results to a CSV file
results_df.to_csv('D:\\Feng_Wenhe\\Python\\Nano_Prediction of depth_Honeywell.csv', index=False)  # Save the results without an index column

# 6. Create Prediction and Error Band Visualization
# Sort predictions and get sorting indices
sorted_indices = np.argsort(results_df['Predicted_D'])

# Rearrange data based on sorted indices
x_index = np.arange(len(results_df))  # Create array of data point indices
y_pred_sorted = results_df['Predicted_D'].iloc[sorted_indices].values  # Sort predicted values
error_range_sorted = results_df['RMSE'].iloc[sorted_indices].values  # Sort error ranges

# Create visualization
plt.figure(figsize=(20, 12))  # Set large figure size for better visibility

# Plot sorted prediction values
plt.plot(x_index, 
         y_pred_sorted, 
         '-', 
         color='blue', 
         label='Predicted D (Sorted)', 
         markersize=5)

# Create error band visualization
plt.fill_between(
    x_index,  # X-axis values
    y_pred_sorted - error_range_sorted,  # Lower bound of error band
    y_pred_sorted + error_range_sorted,  # Upper bound of error band
    color='orange', 
    alpha=0.5,  # Set transparency
    label='Error Range'
)

# Add plot labels and formatting
plt.xlabel('Data Point Index (Sorted)')
plt.ylabel('Predicted Laser Removal Depth')
plt.title('Sorted Predicted Laser Removal Depth with Error Range')
plt.legend()

# Enable grid for better readability
plt.grid(True)

# Display the plot
plt.show()

##############################################
###########Polynomial Regression##############
##############################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# 7. Polynomial Regression Optimization and LOOCV
# Define hyperparameter grid (degree of polynomial)
param_grid_poly = {'polynomialfeatures__degree': [2, 3, 4]}  # Testing 2nd, 3rd, and 4th order polynomials

# Create pipeline for polynomial regression
polyreg_pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())

# Use GridSearchCV to find the best polynomial degree
grid_search_poly = GridSearchCV(polyreg_pipeline, param_grid_poly, scoring='neg_mean_squared_error', cv=5)
grid_search_poly.fit(X_scaled, y_scaled.ravel())

# Retrieve the best model
best_polyreg = grid_search_poly.best_estimator_

# Print best polynomial degree
best_degree_poly = grid_search_poly.best_params_['polynomialfeatures__degree']
print(f'Best Polynomial Degree: {best_degree_poly}')

# Extract polynomial regression coefficients
poly_features = PolynomialFeatures(degree=best_degree_poly)
X_poly = poly_features.fit_transform(X_scaled)
poly_model = LinearRegression().fit(X_poly, y_scaled.ravel())
coefficients = poly_model.coef_
intercept = poly_model.intercept_
print(f'Optimized Polynomial Model: Intercept = {intercept}, Coefficients = {coefficients}')

# Generate polynomial equation as a string
feature_names = ['A', 'B', 'C', 'D']
poly_terms = poly_features.get_feature_names_out(feature_names)

equation_terms = [f"{coeff:.4f}*{term}" for coeff, term in zip(coefficients, poly_terms)]
equation = " + ".join(equation_terms)
equation = f"F = {intercept:.4f} + {equation}"

print("Optimized Polynomial Equation:")
print(equation)


# LOOCV for Polynomial Regression
rmse_scores_polyreg = []
rrmse_scores_polyreg = []
y_pred_list_polyreg = []
y_true_list_polyreg = []

for train_index, test_index in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    
    # Train polynomial regression model
    best_polyreg.fit(X_train, y_train.ravel())
    
    # Predict
    y_pred_scaled_polyreg = best_polyreg.predict(X_test).reshape(-1, 1)
    
    # Inverse transform
    y_pred_value_polyreg = scaler_y.inverse_transform(y_pred_scaled_polyreg)
    y_true_value_polyreg = scaler_y.inverse_transform(y_test)
    
    # Store results
    y_pred_list_polyreg.append(y_pred_value_polyreg[0][0])
    y_true_list_polyreg.append(y_true_value_polyreg[0][0])
    
    # Compute RMSE
    rmse_polyreg = np.sqrt(mean_squared_error(y_true_value_polyreg, y_pred_value_polyreg))
    rmse_scores_polyreg.append(rmse_polyreg)
    
    # Compute relative RMSE (rRMSE)
    rrmse_polyreg = rmse_polyreg / y_range
    rrmse_scores_polyreg.append(rrmse_polyreg)

# Compute final metrics
mean_rmse_polyreg = np.mean(rmse_scores_polyreg)
mean_rrmse_polyreg = np.mean(rrmse_scores_polyreg)
r2_polyreg = r2_score(y_true_list_polyreg, y_pred_list_polyreg)

print(f'LOOCV RMSE (Polynomial Regression): {mean_rmse_polyreg}')
print(f'LOOCV Relative RMSE (Polynomial Regression): {mean_rrmse_polyreg}')
print(f'LOOCV R² (Polynomial Regression): {r2_polyreg:.4f}')

### 8. Plot Comparison of ANN and Polynomial Regression
sorted_indices = np.argsort(y_true_list)

sorted_actual = np.array(y_true_list)[sorted_indices]
sorted_predicted_ann = np.array(y_pred_list)[sorted_indices]
sorted_predicted_polyreg = np.array(y_pred_list_polyreg)[sorted_indices]

plt.figure(figsize=(8, 6))
plt.scatter(range(len(sorted_actual)), sorted_actual, color='blue', alpha=0.6, label='Actual F')
plt.scatter(range(len(sorted_predicted_ann)), sorted_predicted_ann, color='orange', alpha=0.6, label='ANN Predicted F')
plt.scatter(range(len(sorted_predicted_polyreg)), sorted_predicted_polyreg, color='green', alpha=0.6, label='Polynomial Regression Predicted F')

plt.xlabel('Data Points (Sorted)')
plt.ylabel('F Values')
plt.title('Comparison of ANN vs. Polynomial Regression Predictions')
plt.legend()
plt.grid(True)
plt.show()
