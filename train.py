import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from scipy.stats import randint, uniform
import joblib

df = pd.read_csv('school_need_factor_dataset.csv')

X = df.drop(['school_id', 'needFactor'], axis=1)
y = df['needFactor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['area']
numerical_features = ['student_count', 'book_count']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


def create_features(X):
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
        start_idx = len(X_df.columns)
        X_df[start_idx] = X_df[1] / X_df[0] if X_df.shape[1] > 1 else 0
        return X_df.values
    else:
        X_new = X.copy()
        X_new['books_per_student'] = X_new['book_count'] / X_new['student_count']
        return X_new


feature_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_engineering', FunctionTransformer(create_features, validate=False))
])

models = {
    'random_forest': RandomForestRegressor(random_state=42),
    'gradient_boosting': GradientBoostingRegressor(random_state=42),
    'xgboost': xgb.XGBRegressor(random_state=42),
    'elastic_net': ElasticNet(random_state=42)
}

param_distributions = {
    'random_forest': {
        'n_estimators': randint(50, 500),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    },
    'gradient_boosting': {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.7, 0.3)
    },
    'xgboost': {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    },
    'elastic_net': {
        'alpha': uniform(0.001, 1.0),
        'l1_ratio': uniform(0, 1),
        'max_iter': randint(1000, 3000)
    }
}

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions={'model__' + key: value for key, value in param_distributions[name].items()},
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_params': random_search.best_params_,
        'model': best_model
    }

    print(f"\n{name.upper()} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Best Parameters: {random_search.best_params_}")

best_model_name = min(results, key=lambda x: results[x]['rmse'])
print(f"\nBest Model: {best_model_name.upper()} with RMSE: {results[best_model_name]['rmse']:.4f}")

feature_importances = None
if best_model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
    model = results[best_model_name]['model']
    if best_model_name == 'xgboost':
        feature_importances = model.named_steps['model'].feature_importances_
    else:
        feature_importances = model.named_steps['model'].feature_importances_

    feature_names = (
            [f'area_{cat}' for cat in X_train['area'].unique()] +
            ['student_count', 'book_count']
    )

    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importances)],
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

joblib.dump(results[best_model_name]['model'], f'{best_model_name}_needfactor_model.pkl')


def predict_needfactor(area, student_count, book_count):
    data = pd.DataFrame({
        'area': [area],
        'student_count': [student_count],
        'book_count': [book_count]
    })
    return results[best_model_name]['model'].predict(data)[0]


print("\nPrediction Examples:")
test_cases = [
    ('Rural', 200, 50),
    ('Urban', 500, 2000),
    ('Metropolitan', 1500, 9000)
]

for area, students, books in test_cases:
    prediction = predict_needfactor(area, students, books)
    print(f"Area: {area}, Students: {students}, Books: {books} → Predicted Need Factor: {prediction:.2f}")