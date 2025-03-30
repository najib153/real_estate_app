from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_linear_regression(X_train, y_train):
    """Train and evaluate Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(train_pred, y_train)
    return model, train_mae

def train_decision_tree(X_train, y_train, max_depth=3, max_features=10, random_state=567):
    """Train and evaluate Decision Tree model"""
    model = DecisionTreeRegressor(
        max_depth=max_depth, 
        max_features=max_features, 
        random_state=random_state)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(train_pred, y_train)
    return model, train_mae

def train_random_forest(X_train, y_train, n_estimators=200, criterion='absolute_error'):
    """Train and evaluate Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        criterion=criterion)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(train_pred, y_train)
    return model, train_mae

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data"""
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(test_pred, y_test)
    return test_mae