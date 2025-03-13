
# Import necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    try:
        print("Diabetes Prediction Models Analysis")
        
        # Load the dataset
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        
        # Convert to DataFrame for easier data exploration
        feature_names = diabetes.feature_names
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Display basic information about the dataset
        print("Dataset Information:")
        print(f"Number of samples: {df.shape[0]}")
        print(f"Number of features: {len(feature_names)}")
        print("\nFeature names:")
        for name in feature_names:
            print(f"- {name}")
        
        # Data exploration
        print("\nData Summary:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Create output directory for results if it doesn't exist
        try:
            os.makedirs('results', exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create results directory: {str(e)}")
        
        # Split the data into training and testing sets (80/20 split)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error during data splitting: {str(e)}")
            return
        
        # Standardize the features
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Error during feature scaling: {str(e)}")
            return
        
        # Function to evaluate model performance
        def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                train_rmse = np.sqrt(train_mse)
                test_rmse = np.sqrt(test_mse)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                # Print the results
                print(f"\n{model_name} Performance:")
                print(f"Training RMSE: {train_rmse:.2f}")
                print(f"Testing RMSE: {test_rmse:.2f}")
                print(f"Training MAE: {train_mae:.2f}")
                print(f"Testing MAE: {test_mae:.2f}")
                print(f"Training R²: {train_r2:.4f}")
                print(f"Testing R²: {test_r2:.4f}")
                print(f"5-Fold CV RMSE: {cv_rmse:.2f}")
                
                # Save predictions to CSV
                try:
                    pred_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': y_pred_test
                    })
                    pred_df.to_csv(f'results/{model_name.replace(" ", "_").lower()}_predictions.csv', index=False)
                except Exception as e:
                    print(f"Warning: Could not save predictions to CSV: {str(e)}")
                
                return {
                    'model': model,
                    'name': model_name,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_rmse': cv_rmse
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                return None
        
        # Train and evaluate models
        models_results = []
        
        try:
            # Model 1: Linear Regression
            linear_model = LinearRegression()
            linear_results = evaluate_model(linear_model, "linear_regression", X_train_scaled, X_test_scaled, y_train, y_test)
            if linear_results:
                models_results.append(linear_results)
        except Exception as e:
            print(f"Error with Linear Regression model: {str(e)}")
        
        try:
            # Model 2: Random Forest Regressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_results = evaluate_model(rf_model, "random_forest", X_train_scaled, X_test_scaled, y_train, y_test)
            if rf_results:
                models_results.append(rf_results)
        except Exception as e:
            print(f"Error with Random Forest model: {str(e)}")
        
        try:
            # Model 3: Support Vector Regression
            svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
            svr_results = evaluate_model(svr_model, "support_vector_regression", X_train_scaled, X_test_scaled, y_train, y_test)
            if svr_results:
                models_results.append(svr_results)
        except Exception as e:
            print(f"Error with SVR model: {str(e)}")
        
        # Compare the models if we have results
        if not models_results:
            print("No models were successfully evaluated. Exiting.")
            return
        
        # Create a comparison table
        try:
            comparison_data = {
                'Model': [model['name'] for model in models_results],
                'Training RMSE': [model['train_rmse'] for model in models_results],
                'Testing RMSE': [model['test_rmse'] for model in models_results],
                'Training MAE': [model['train_mae'] for model in models_results],
                'Testing MAE': [model['test_mae'] for model in models_results],
                'Training R²': [model['train_r2'] for model in models_results],
                'Testing R²': [model['test_r2'] for model in models_results],
                'CV RMSE': [model['cv_rmse'] for model in models_results]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            print("\nModel Comparison:")
            print(comparison_df)
            
            # Save comparison to CSV
            comparison_df.to_csv('results/model_comparison.csv', index=False)
        except Exception as e:
            print(f"Error creating model comparison: {str(e)}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main()

# Best performing model analysis
# SVR is a good model because it ranks first in 3 out of 4 testing metrics (RMSE, MAE, and R²), even though it has the worst CV RMSE.
# SVR has the lowest RMSE and MAE, and the highest R², indicating better performance on the test set compared to other models.
# However, it has the worst CV RMSE, which indicates the model is overfitting, meaning it performs well on training data but struggles to generalize to new, 
# unseen data, as shown by poor performance during cross-validation. 
# Linear regression performs decently with the Testing RMSE and MAE, though not as well as SVR.
# The Linear regression CV RMSE is lower than SVR’s, meaning the model performs better when tested on different data during cross-validation. 
# This shows it generalizes better( it's more reliable on new, unseen data).
# Random Forest does poorly across most metrics, especially Testing RMSE, Testing MAE, and Testing R², 
# compared to both SVR and Linear Regression. It performs the worst in almost all categories.
# Overall, Linear Regression does the best since it performs more consistently across all metrics, including CV RMSE.
