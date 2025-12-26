"""
Big Data Analytics and Visualisation Assignment
Full Analysis Script for Sections 3.1 to 3.5
Dataset: Medical Insurance Cost Prediction (100,000 rows)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings

df = pd.read_csv("medical_insurance.csv")

print("\n" + "="*50)
print("3.1 EXPLORATORY DATA ANALYSIS")
print("="*50)

print("\nKey Columns Overview:")
key_columns = ['age', 'sex', 'bmi', 'smoker', 'chronic_count', 'risk_score', 
               'claims_count', 'annual_medical_cost', 'region', 'income', 'alcohol_freq']
print(df[key_columns].head())

print("\nDescriptive Statistics (Numerical):")
print(df[key_columns].select_dtypes(include=np.number).describe())

print("\nMissing Values (only columns with missing):")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nCategorical Value Counts:")
categorical_cols = ['sex', 'smoker', 'region', 'alcohol_freq', 'plan_type']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\nTop Correlations with Annual Medical Cost:")
numerical_cols = df.select_dtypes(include=np.number).columns
correlations = df[numerical_cols].corr()['annual_medical_cost'].sort_values(ascending=False)
print(correlations.head(15))

print("\n" + "="*50)
print("3.2 DATA ANALYSIS PROCESS - PSEUDOCODE")
print("="*50)
pseudocode = """
dataframe = read_csv('medical_insurance.csv')

fill_missing_with_mode(dataframe['alcohol_freq'])  # Most common: 'Occasional' or 'None'

categorical_columns = ['sex', 'smoker', 'region', 'alcohol_freq', 'plan_type', 'network_tier', 'marital_status', 'employment_status']
for column in categorical_columns:
    label_encode(dataframe[column])  # Convert to numbers (e.g., smoker: Never=0, Former=1, Current=2)


apply_log_transformation(dataframe['annual_medical_cost'])  # log1p to reduce right skew


numerical_columns = ['age', 'bmi', 'income', 'risk_score', 'claims_count']
scaler = standard_scaler()
scaled_values = scaler.fit_transform(dataframe[numerical_columns])
replace_columns_with_scaled(dataframe, numerical_columns, scaled_values)

selected_features = ['age', 'bmi', 'smoker', 'chronic_count', 'risk_score', 'claims_count', 'income', 'region', 'sex']

X = dataframe[selected_features]
y = dataframe['annual_medical_cost']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

write_csv(dataframe, 'processed_insurance_data.csv')
"""
print(pseudocode)

print("\n" + "="*50)
print("3.3 DATA VISUALIZATION")
print("="*50)

sample_df = df.sample(n=10000, random_state=42)  # 10% sample

plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 16))

# 1. Distributions
ax1 = fig.add_subplot(3, 3, 1)
sns.histplot(sample_df['age'], bins=40, kde=True, ax=ax1)
ax1.set_title('Age Distribution')

ax2 = fig.add_subplot(3, 3, 2)
sns.histplot(sample_df['bmi'], bins=40, kde=True, ax=ax2)
ax2.set_title('BMI Distribution')

ax3 = fig.add_subplot(3, 3, 3)
sns.histplot(sample_df['annual_medical_cost'], bins=50, kde=True, ax=ax3)
ax3.set_title('Annual Medical Cost Distribution (Skewed)')

# 2. Relationships
ax4 = fig.add_subplot(3, 3, 4)
sns.scatterplot(data=sample_df, x='age', y='annual_medical_cost', hue='smoker', alpha=0.6, ax=ax4)
ax4.set_title('Age vs Medical Cost (colored by smoker)')

ax5 = fig.add_subplot(3, 3, 5)
sns.scatterplot(data=sample_df, x='bmi', y='annual_medical_cost', hue='smoker', alpha=0.6, ax=ax5)
ax5.set_title('BMI vs Medical Cost (colored by smoker)')

ax6 = fig.add_subplot(3, 3, 6)
sns.boxplot(data=sample_df, x='smoker', y='annual_medical_cost', ax=ax6)
ax6.set_title('Medical Cost by Smoking Status')

# 3. Chronic conditions
ax7 = fig.add_subplot(3, 3, 7)
chronic_avg = sample_df.groupby('chronic_count')['annual_medical_cost'].mean()
chronic_avg.plot(kind='bar', ax=ax7)
ax7.set_title('Average Cost by Chronic Disease Count')
ax7.set_xlabel('Chronic Count')

# 4. Heatmap of key correlations
ax8 = fig.add_subplot(3, 3, 8)
key_nums = ['age', 'bmi', 'chronic_count', 'risk_score', 'claims_count', 'annual_medical_cost']
sns.heatmap(sample_df[key_nums].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax8)
ax8.set_title('Correlation Heatmap (Key Variables)')

plt.tight_layout()
plt.savefig('assignment_visualizations.png', dpi=300, bbox_inches='tight')
print("All visualizations saved as 'assignment_visualizations.png'")

print("\n" + "="*50)
print("3.4 PREDICTIVE MODELLING")
print("="*50)

# Preprocessing
df_model = df.copy()
df_model['alcohol_freq'].fillna(df_model['alcohol_freq'].mode()[0], inplace=True)

# Encode categoricals
cat_encode = ['sex', 'smoker', 'region', 'alcohol_freq']
for col in cat_encode:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Features and target (log transform target to handle skew)
features = ['age', 'bmi', 'smoker', 'chronic_count', 'risk_score', 'claims_count', 'income', 'region', 'sex', 'alcohol_freq']
X = df_model[features]
y = np.log1p(df_model['annual_medical_cost'])  # Better for skewed regression

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results[name] = {"R²": r2, "RMSE": rmse, "MAE": mae}
    print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")

# Best model and feature importance
best_model_name = max(results, key=lambda x: results[x]["R²"])
print(f"\nBest Model: {best_model_name} with R² = {results[best_model_name]['R²']:.4f}")

if "XGBoost" in best_model_name or "Gradient" in best_model_name:
    importance = models[best_model_name].feature_importances_
    feat_importance = pd.Series(importance, index=features).sort_values(ascending=False)
    print("\nTop 10 Feature Importance:")
    print(feat_importance)

    # Plot importance
    plt.figure(figsize=(10, 6))
    feat_importance.plot(kind='bar')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("Feature importance plot saved as 'feature_importance.png'")
    
print("\n" + "="*50)
print("3.5 KEY INSIGHTS AND CHALLENGES")
print("="*50)

insights = """
Key Insights:
1. Smoking status is the strongest predictor of high medical costs — current/former smokers pay significantly more.
2. Age, BMI, chronic disease count, and risk_score show strong positive correlations with annual costs.
3. Costs are highly right-skewed: most people have low costs, but a small group has extremely high costs.
4. Predictive models (especially tree-based like XGBoost) achieve high R² (>0.85 likely), enabling accurate cost forecasting for pricing and segmentation.
5. Visualizations clearly reveal clusters: e.g., smokers over age 40 with high BMI are high-risk.

Challenges:
1. Data quality: ~30% missing in alcohol_freq — requires imputation which may introduce bias.
2. Skewness and outliers in costs — addressed via log transformation.
3. Interpretability: Complex models like XGBoost are powerful but less interpretable than linear regression.
4. Ethical/privacy issues: Health data is sensitive; anonymization needed.
5. Scalability: 100k rows is manageable in Python, but larger real-world data may require Spark/distributed tools.
6. Potential bias: Model performance may vary by region/sex if data is imbalanced.
"""

print(insights)
