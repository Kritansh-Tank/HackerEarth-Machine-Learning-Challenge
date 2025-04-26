import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Loading data...")
# Load the data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')
sample_submission = pd.read_csv('dataset/sample_submission.csv')

# Create a copy of the datasets
train_df = train_data.copy()
test_df = test_data.copy()

# Print dataset info
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

# Check the format of the sample submission
print("Sample submission format:")
print(sample_submission.head())

# Define target variable and features
target = 'carbon_footprint'
id_column = 'ID'

print("Preprocessing data...")
# Data cleaning and preprocessing function


def preprocess_data(df, is_train=True):
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # Identify numeric and categorical columns
    if is_train:
        numeric_cols = df_clean.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        if id_column in numeric_cols:
            numeric_cols.remove(id_column)
        categorical_cols = df_clean.select_dtypes(
            include=['object']).columns.tolist()
        if id_column in categorical_cols:
            categorical_cols.remove(id_column)
    else:
        numeric_cols = [col for col in df_clean.columns if col !=
                        id_column and df_clean[col].dtype in ['int64', 'float64']]
        categorical_cols = [col for col in df_clean.columns if col !=
                            id_column and df_clean[col].dtype == 'object']

    # Handle negative values in numeric columns (likely error codes)
    for col in numeric_cols:
        # Replace negative values with NaN (except for specific columns where negative values might be valid)
        if col not in ['public_transport_usage_per_week']:
            neg_mask = pd.to_numeric(df_clean[col], errors='coerce') < 0
            if neg_mask.any():
                df_clean.loc[neg_mask, col] = np.nan

        # Convert to numeric, coercing errors to NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Convert boolean columns to 1/0
    bool_cols = df_clean.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)

    return df_clean, numeric_cols, categorical_cols


# Preprocess the data
train_clean, numeric_features, categorical_features = preprocess_data(train_df)
test_clean, _, _ = preprocess_data(test_df, is_train=False)

print("Engineering features...")
# Feature Engineering


def engineer_features(df, numeric_cols):
    df_eng = df.copy()

    # Handle division by zero or NaN values
    df_eng['household_size'] = df_eng['household_size'].replace(0, np.nan)

    # Create the most important features that are likely to impact carbon footprint

    # Combined energy usage (electricity + gas)
    df_eng['total_energy'] = df_eng['electricity_kwh_per_month'] + \
        (df_eng['natural_gas_therms_per_month']
         * 29.3)  # Convert therms to kWh
    numeric_cols.append('total_energy')

    # Sustainability score - higher is better for environment
    sustainability_factors = [
        'recycles_regularly',
        'composts_organic_waste',
        'uses_solar_panels',
        'energy_efficient_appliances',
        'smart_thermostat_installed'
    ]

    # Create the sustainability score
    df_eng['sustainability_score'] = 0
    for factor in sustainability_factors:
        if factor in df_eng.columns:
            # Fill NaN with 0 for this calculation
            df_eng['sustainability_score'] += df_eng[factor].fillna(0)

    numeric_cols.append('sustainability_score')

    # Dietary impact (based on diet type)
    diet_impact = {
        'vegan': 1,         # Lowest impact
        'vegetarian': 2,    # Medium-low impact
        'omnivore': 3       # Highest impact
    }

    if 'diet_type' in df_eng.columns:
        # Create a new feature for diet impact
        df_eng['diet_impact'] = df_eng['diet_type'].map(diet_impact).fillna(2)
        numeric_cols.append('diet_impact')

    return df_eng, numeric_cols


# Engineer features
train_featured, numeric_features = engineer_features(
    train_clean, numeric_features)
test_featured, _ = engineer_features(test_clean, numeric_features)

# Separate features and target
X = train_featured.drop([target, id_column], axis=1)
y = train_featured[target]
test_X = test_featured.drop([id_column], axis=1)

print("Setting up model pipeline...")
# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Use a single strong model instead of stacking
# XGBoost is often the best single model choice for tabular data
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    ))
])

# Split data for model evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

print("Training model...")
# Train the model
model.fit(X_train, y_train)

# Make predictions on validation set
val_predictions = model.predict(X_val)

# Evaluate model performance
r2 = r2_score(y_val, val_predictions)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
mae = mean_absolute_error(y_val, val_predictions)

print(f"\nModel Performance on Validation Set:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Scaled Competition Score: {max(0, 100 * r2):.2f}")

# Cross-validation with fewer folds for speed
print("\nPerforming 3-fold cross-validation...")
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
scaled_cv_scores = [max(0, 100 * score) for score in cv_scores]

print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R² Score: {np.mean(cv_scores):.4f}")
print(f"Mean Scaled Competition Score: {np.mean(scaled_cv_scores):.2f}")

print("\nTraining final model on all data...")
# Train final model on all data
model.fit(X, y)

print("Making predictions on test data...")
# Make predictions on test data
test_predictions = model.predict(test_X)

# Create submission DataFrame with the correct IDs from test_df
submission = pd.DataFrame({
    id_column: test_df[id_column],
    target: test_predictions
})

# Make sure columns match sample submission format
print("\nVerifying submission format:")
print(f"Sample submission columns: {sample_submission.columns.tolist()}")
print(f"Our submission columns: {submission.columns.tolist()}")

# Check if our column order matches the sample submission
if sample_submission.columns.tolist() != submission.columns.tolist():
    print("Reordering columns to match sample submission format...")
    submission = submission[sample_submission.columns.tolist()]

print(f"Submission shape: {submission.shape}")
print(f"Expected shape based on test data: {test_df.shape[0]} rows, 2 columns")

# Save the submission file
submission.to_csv('output/submission.csv', index=False)
print("Submission file created!")

# Display submission statistics
print("\nSubmission Statistics:")
print(f"Count: {len(submission)}")
print(f"Mean: {submission['carbon_footprint'].mean():.2f}")
print(f"Min: {submission['carbon_footprint'].min():.2f}")
print(f"Max: {submission['carbon_footprint'].max():.2f}")
print(f"Std Dev: {submission['carbon_footprint'].std():.2f}")

# Try to create a simple visualization
try:
    plt.figure(figsize=(10, 5))
    sns.histplot(test_predictions, kde=True)
    plt.title('Test Predictions Distribution')
    plt.savefig('output/prediction_distribution.png')
    print("Prediction distribution plot saved.")
except Exception as e:
    print(f"Could not create visualization: {e}")
