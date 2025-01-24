import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
 
# Load and Preprocess Data
def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path, sep=';')
   
    # Remove unnecessary columns
    df = df.drop('id', axis=1)
   
    # Imputation and Encoding
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
   
    # Impute numeric columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
   
    # Encode categorical columns
    label_encoders = {}
    for col in non_numeric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
   
    return df, label_encoders
 
# Advanced Model Training
def train_advanced_model(X, y, model_type='flood'):
    # Apply SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
   
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_resampled, y_resampled)
   
    # Hyperparameter grid
    if model_type == 'flood':
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        base_model = RandomForestClassifier(random_state=42)
    else:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10]
        }
        base_model = GradientBoostingClassifier(random_state=42)
   
    # Randomized Search
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        random_state=42,
        verbose=1
    )
   
    # Fit the model
    random_search.fit(X_selected, y_resampled)
   
    return random_search.best_estimator_, selector
 
# Visualization Functions
def plot_feature_importance(model, feature_names, title):
    plt.figure(figsize=(12, 8))
    feature_imp = model.feature_importances_
    indices = np.argsort(feature_imp)
    plt.title(title)
    plt.barh(range(len(indices)), feature_imp[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()
 
def plot_prediction_comparison(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title('True Values')
    plt.hist(y_true, bins=[0, 0.5, 1, 1.5], color='blue', alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.title('Predicted Values')
    plt.hist(y_pred, bins=[0, 0.5, 1, 1.5], color='red', alpha=0.7)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
 
# Main Execution
def main():
    # Load training data
    file_path = r"C:\Users\nino\Documents\YNOV\Hackathon\Hackaton_24_25\catastrophes_naturelles_vf.csv"
    df, label_encoders = load_and_preprocess_data(file_path)
   
    # Prepare features and targets
    features = df.drop(columns=['Innondation', 'seisme'])
    target_inondation = df['Innondation']
    target_seisme = df['seisme']
   
    # Split data
    X_train_inondation, X_test_inondation, y_train_inondation, y_test_inondation = train_test_split(
        features, target_inondation, test_size=0.2, random_state=42
    )
    X_train_seisme, X_test_seisme, y_train_seisme, y_test_seisme = train_test_split(
        features, target_seisme, test_size=0.2, random_state=42
    )
   
    # Standardize features
    scaler = StandardScaler()
    X_train_inondation = scaler.fit_transform(X_train_inondation)
    X_test_inondation = scaler.transform(X_test_inondation)
    X_train_seisme = scaler.fit_transform(X_train_seisme)
    X_test_seisme = scaler.transform(X_test_seisme)
   
    # Train models
    flood_model, flood_selector = train_advanced_model(X_train_inondation, y_train_inondation, 'flood')
    earthquake_model, earthquake_selector = train_advanced_model(X_train_seisme, y_train_seisme, 'earthquake')
   
    # Predictions on test set
    y_pred_inondation = flood_model.predict(flood_selector.transform(X_test_inondation))
    y_pred_seisme = earthquake_model.predict(earthquake_selector.transform(X_test_seisme))
   
    # Evaluate models
    print("Flood Model Performance:")
    print(classification_report(y_test_inondation, y_pred_inondation))
    print("\nEarthquake Model Performance:")
    print(classification_report(y_test_seisme, y_pred_seisme))
   
    # Visualizations
    plot_feature_importance(flood_model, features.columns, "Feature Importance - Flood Model")
    plot_prediction_comparison(y_test_inondation, y_pred_inondation, "Flood Prediction Comparison")
    plot_prediction_comparison(y_test_seisme, y_pred_seisme, "Earthquake Prediction Comparison")
 
if __name__ == "__main__":
    #main()