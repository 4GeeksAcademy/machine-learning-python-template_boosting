from utils import db_connect
engine = db_connect()

# app.py

# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import xgboost as xgb
import shap

# Crear directorios necesarios
os.makedirs('models', exist_ok=True)

# Definir rutas de los archivos
train_file_path = 'data/processed/clean_train.csv'
test_file_path = 'data/processed/clean_test.csv'

# Verificar si los archivos existen
if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
    raise FileNotFoundError(f"Los archivos {train_file_path} y/o {test_file_path} no existen. Asegúrate de que los archivos están en la ruta correcta.")

# Cargar datos procesados
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Separar características y la variable objetivo
X_train = train_data.drop('Outcome', axis=1)
y_train = train_data['Outcome']
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

# Entrenar y evaluar el modelo de Árbol de Decisión
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_dt = decision_tree_model.predict(X_test)
y_pred_dt_proba = decision_tree_model.predict_proba(X_test)[:, 1]
decision_tree_results = {
    "Accuracy": accuracy_score(y_test, y_pred_dt),
    "ROC AUC Score": roc_auc_score(y_test, y_pred_dt_proba),
    "F1 Score": f1_score(y_test, y_pred_dt),
    "Precision": precision_score(y_test, y_pred_dt),
    "Recall": recall_score(y_test, y_pred_dt)
}
joblib.dump(decision_tree_model, 'models/best_decision_tree_model.pkl')

# Entrenar y evaluar el modelo de Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_rf_proba = random_forest_model.predict_proba(X_test)[:, 1]
random_forest_results = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "ROC AUC Score": roc_auc_score(y_test, y_pred_rf_proba),
    "F1 Score": f1_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf)
}
joblib.dump(random_forest_model, 'models/best_random_forest_model.pkl')

# Construir el modelo de XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgboost_results = {
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "ROC AUC Score": roc_auc_score(y_test, y_pred_xgb_proba),
    "F1 Score": f1_score(y_test, y_pred_xgb),
    "Precision": precision_score(y_test, y_pred_xgb),
    "Recall": recall_score(y_test, y_pred_xgb)
}
joblib.dump(xgb_model, 'models/best_xgboost_model.pkl')

# Imprimir resultados de XGBoost
print("XGBoost Results")
print(xgboost_results)
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))

# Comparar resultados
all_models_results = {
    'Decision Tree': decision_tree_results,
    'Random Forest': random_forest_results,
    'XGBoost': xgboost_results
}

# Mostrar los resultados en un DataFrame
all_models_results_df = pd.DataFrame(all_models_results).T
print(all_models_results_df)

# Visualización de los resultados
all_models_results_df.plot(kind='bar', figsize=(14, 10))
plt.title('Comparación de Modelos: Árbol de Decisión, Random Forest y XGBoost')
plt.ylabel('Score')
plt.show()

# Interpretación de los Modelos con SHAP
# Crear un explicador SHAP para Random Forest
explainer_rf = shap.Explainer(random_forest_model)
shap_values_rf = explainer_rf(X_test)
shap.summary_plot(shap_values_rf, X_test, feature_names=X_test.columns)

# Crear un explicador SHAP para XGBoost
explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_test)
shap.summary_plot(shap_values_xgb, X_test, feature_names=X_test.columns)

# Conclusión final
print("El mejor modelo para este conjunto de datos es el Random Forest debido a su mejor desempeño en términos de precisión, ROC AUC Score, F1 Score y balance entre precisión y recall.")