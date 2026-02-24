"""
Study Time Recommendation - Model Training Script
Run this to train and save your ML model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("=" * 60)
print("STUDY TIME RECOMMENDATION - MODEL TRAINING")
print("=" * 60)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Step 1: Load the dataset
print("\n📂 Loading dataset...")
# Find the most recent dataset file
dataset_files = [f for f in os.listdir('datasets') if f.startswith('study_time_dataset') and f.endswith('.csv')]
if dataset_files:
    latest_dataset = sorted(dataset_files)[-1]
    df = pd.read_csv(f'datasets/{latest_dataset}')
    print(f"✅ Dataset loaded: {latest_dataset}")
else:
    # Try to load from current directory
    df = pd.read_csv('study_time_dataset.csv')
    print("✅ Dataset loaded from current directory")

print(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Exploratory Data Analysis (EDA)
print("\n🔍 Performing Exploratory Data Analysis...")

# Display basic info
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

# Check for missing values
print("\n⚠️ Missing Values:")
print(df.isnull().sum())

# Correlation analysis
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('models/correlation_matrix.png')
plt.close()
print("✅ Correlation matrix saved to models/correlation_matrix.png")

# Step 3: Feature Selection
print("\n🎯 Selecting features for training...")

# Define features (X) and target (y)
feature_columns = ['gpa', 'difficulty_level', 'past_performance', 'available_hours']
target_column = 'recommended_study_hours'

# Verify columns exist
available_features = [col for col in feature_columns if col in df.columns]
if len(available_features) < len(feature_columns):
    missing = set(feature_columns) - set(available_features)
    print(f"⚠️ Warning: Missing columns: {missing}")
    feature_columns = available_features

X = df[feature_columns]
y = df[target_column]

print(f"✅ Features: {feature_columns}")
print(f"✅ Target: {target_column}")
print(f"📐 X shape: {X.shape}")
print(f"📐 y shape: {y.shape}")

# Step 4: Split the data
print("\n✂️ Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Step 5: Scale the features
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved to models/scaler.pkl")

# Step 6: Train multiple models and compare
print("\n🤖 Training multiple models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n📈 Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  ✅ R² Score: {r2:.4f}")
    print(f"  ✅ RMSE: {rmse:.4f}")
    print(f"  ✅ MAE: {mae:.4f}")
    print(f"  ✅ CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Step 7: Compare models
print("\n📊 Model Comparison:")
comparison_df = pd.DataFrame({
    name: {
        'R² Score': results[name]['r2'],
        'RMSE': results[name]['rmse'],
        'MAE': results[name]['mae'],
        'CV R² Mean': results[name]['cv_mean']
    }
    for name in results
}).T

print(comparison_df.round(4))

# Save comparison to CSV
comparison_df.to_csv('models/model_comparison.csv')
print("✅ Model comparison saved to models/model_comparison.csv")

# Step 8: Select the best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")

# Step 9: Hyperparameter tuning for the best model
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print(f"\n🔧 Performing hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42)
    else:  # Gradient Boosting
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        base_model = GradientBoostingRegressor(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, 
        scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # Evaluate tuned model
    y_pred_tuned = best_model.predict(X_test_scaled)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    print(f"✅ Tuned model R² on test set: {r2_tuned:.4f}")

# Step 10: Save the best model
model_filename = f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Best model saved to {model_filename}")

# Also save as 'latest_model.pkl' for easy reference
joblib.dump(best_model, 'models/latest_model.pkl')
print("✅ Model also saved as models/latest_model.pkl")

# Step 11: Feature Importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    print("\n🌟 Feature Importance:")
    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()
    print("✅ Feature importance plot saved to models/feature_importance.png")

# Step 12: Test predictions
print("\n🧪 Testing model with sample predictions:")
sample_inputs = [
    [3.8, 2, 85, 5],   # Good student, easy subject
    [2.5, 4, 60, 3],   # Struggling student, hard subject
    [3.2, 5, 70, 2],   # Limited time, hard subject
    [3.5, 3, 80, 4],   # Average student
    [2.8, 4, 55, 6]    # Poor performance, more time available
]

for i, sample in enumerate(sample_inputs):
    sample_scaled = scaler.transform([sample])
    prediction = best_model.predict(sample_scaled)[0]
    prediction = round(prediction * 2) / 2  # Round to nearest 0.5
    print(f"  Sample {i+1}: {sample} → Recommended: {prediction} hours")

# Step 13: Save model metadata
metadata = {
    'model_name': best_model_name,
    'features': str(feature_columns),
    'target': target_column,
    'r2_score': results[best_model_name]['r2'],
    'rmse': results[best_model_name]['rmse'],
    'mae': results[best_model_name]['mae'],
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

metadata_df = pd.DataFrame([metadata])
metadata_df.to_csv('models/model_metadata.csv', index=False)
print("\n✅ Model metadata saved to models/model_metadata.csv")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\n📁 Files created in 'models' directory:")
print("   - scaler.pkl")
print("   - latest_model.pkl")
print(f"   - {best_model_name.lower().replace(' ', '_')}_model.pkl")
print("   - model_comparison.csv")
print("   - model_metadata.csv")
print("   - correlation_matrix.png")
print("   - feature_importance.png")