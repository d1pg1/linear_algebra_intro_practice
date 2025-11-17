# Please, compare and analyze results. Add conclusions as comments here or to a readme file.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=666)

    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    model = Ridge(random_state=42)
    parameters = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(model, parameters, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    model = Lasso(random_state=42)
    parameters = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(model, parameters, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression(penalty=None, random_state=42)
    model.fit(X, y)
    return model

def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    model = LogisticRegression(penalty='l2', random_state=42)
    parameters = {'C': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(model, parameters, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    parameters = {'C': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(model, parameters, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def regression_performance_check(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(model_name, "MSE:", mse, "R2:", r2)

def classification_performance_check(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(model_name, "Accuracy:", acc, "Precision:", prec, "Recall:", rec, "F1-Score:", f1)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_regression_data()

    regression_performance_check("Linear Regression", linear_regression(X_train, y_train), X_test, y_test)
    regression_performance_check("Ridge  Regression", ridge_regression(X_train, y_train), X_test, y_test)
    regression_performance_check("Lasso  Regression", lasso_regression(X_train, y_train), X_test, y_test)

    X_train, X_test, y_train, y_test = get_classification_data()

    classification_performance_check("Logistic    Regression", logistic_regression(X_train, y_train), X_test, y_test)
    classification_performance_check("Logistic L1 Regression", logistic_l1_regression(X_train, y_train), X_test, y_test)
    classification_performance_check("Logistic L2 Regression", logistic_l2_regression(X_train, y_train), X_test, y_test)

# Regression conclusions

# All models show moderate predictive power (R2 ~ 0.52–0.53), 
# but Lasso achieves the best performance, with the lowest MSE and highest R2, 
# indicating that feature selection improves generalization. 
# Ridge slightly outperforms Linear Regression, confirming that regularization helps stabilize the model, 
# but Lasso provides the most effective balance between bias and variance on this dataset.

#Classification conclusions

# All three logistic models perform very well, but Logistic Regression with L2 regularization is the strongest overall, 
# achieving the highest accuracy (0.982) and the best F1-score (0.985) while reaching perfect recall (1.0).
# The L1 model also performs strongly and more balanced than the plain model, 
# but the L2-regularized model offers the best generalization, capturing all positives while maintaining high precision.