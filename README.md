# Diabetes-Prediction-Machine-Learning

Learned how to use Scikit-learn and develop multiple regression models.

## Diabetes Prediction Using Machine Learning Models

### Overview
This project aims to predict diabetes progression using machine learning regression models. Utilizing Scikit-learn's built-in diabetes dataset, which contains various medical measurements, the goal is to predict the disease progression over time. Three different machine learning models - Linear Regression, Random Forest Regressor, and Support Vector Regression (SVR) - are developed and evaluated using multiple performance metrics. The best-performing model is identified and analyzed in detail.

### Purpose
The purpose of this project is to use machine learning techniques to predict diabetes progression based on the provided dataset. By leveraging regression models, we aim to improve the accuracy of diabetes outcome predictions, which can be useful for early medical interventions. The models are evaluated using several performance metrics such as RMSE, MAE, R² score, and cross-validation RMSE to determine their effectiveness in real-world applications.

### Files
- **`regsk11.py`**: Python code implementing the models.
- **Copilotchat_11**: Copilot help 

### Dataset
The dataset used is the Diabetes dataset from Scikit-learn. It includes the following:

- **Features**: Various medical measurements related to diabetes progression.
- **Target**: A continuous variable indicating the diabetes progression.

### Libraries Used
- `Numpy` as `np`
- `Pandas` as `pd`
- `Sklearn`

### Sklearn Modules Used
- **datasets**: Provides built-in datasets.
- **metrics**: Performance evaluation metrics.
- **model_selection**: Splits data into training and testing.
- **preprocessing**: Standardizes and scales the data.
- **linear_model**: Implements regression models.
- **ensemble**: Combines multiple models.
- **svm**: Implements Support Vector Regression.

### Models Used
- **Linear Regression**: Predicts outcomes based on a linear relationship between features.
- **Random Forest Regressor**: Uses multiple decision trees to improve prediction accuracy.
- **Support Vector Regression (SVR)**: Uses kernel functions to find complex relationships between features.

### Evaluation Metrics
The following metrics are used to evaluate the models:
- **RMSE (Root Mean Squared Error)**: Measures the model's prediction error.
- **MAE (Mean Absolute Error)**: Measures the average magnitude of errors in predictions.
- **R² Score**: Indicates how well the model explains variance in the target variable.
- **Cross-validation RMSE**: Measures model reliability across different data subsets.

### Project Structure

#### Data Preprocessing:
- The data is split into training (80%) and testing (20%) sets.
- Features are standardized to have a mean of 0 and variance of 1.

#### Model Evaluation:
- Each model is trained on the scaled training data and evaluated on the testing data.
- Model performance is compared using the metrics mentioned above.

#### Model Comparison:
- All models are evaluated and compared to determine the best-performing one.


### Limitations
- The dataset size is relatively small and may not generalize well to all diabetes cases.
- Overfitting Risk Random Forest and Support Vector Regression (SVR) models are prone to overfitting, especially when the dataset is small 

