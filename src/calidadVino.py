import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# 1. Cargar el dataset
df = pd.read_csv('./data/WineQT.csv')

# 2. Exploración de Datos
print("Primeras filas del dataset:")
print(df.head())

# 3. Preprocesamiento de Datos
# 3.1 Identificación de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 3.2 Dividir los datos en variables independientes y dependientes
X = df.drop(columns=['quality', 'Id'])
y = df['quality']

# 3.3 Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3.4 Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entrenamiento de los Modelos de Clasificación

# Modelos a usar
models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Resultados de la evaluación de los modelos
model_results = {}

# Entrenar y evaluar los modelos
for model_name, model in models.items():
    print(f"\nEntrenando el modelo: {model_name}")
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  # Agregar zero_division
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  # Agregar zero_division
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  # Agregar zero_division
    
    model_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Imprimir los resultados de la clasificación
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matriz de confusión
    print(f"Matriz de Confusión:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Informe de Clasificación:\n{classification_report(y_test, y_pred)}")
    
    # Curva ROC y AUC (calculando una curva ROC por cada clase)
    n_classes = len(np.unique(y))
    fpr_all, tpr_all, auc_all = {}, {}, {}
    
    for i in range(n_classes):
        # Verificar si hay al menos una clase positiva en y_test
        if np.any(y_test == i):
            # Curva ROC por clase
            fpr, tpr, thresholds = roc_curve(y_test == i, model.predict_proba(X_test_scaled)[:, i])
            auc = roc_auc_score(y_test == i, model.predict_proba(X_test_scaled)[:, i])
            fpr_all[i], tpr_all[i], auc_all[i] = fpr, tpr, auc
            
            print(f"AUC para la clase {i}: {auc:.2f}")
            
            # Graficar la curva ROC para cada clase
            plt.plot(fpr, tpr, label=f'{model_name} - Clase {i} (AUC = {auc:.2f})')

# 5. Comparación de Resultados
print("\nResultados de todos los modelos:")
for model_name, results in model_results.items():
    print(f"{model_name}: {results}")

# Mostrar la curva ROC de todos los modelos
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Línea diagonal (random classifier)
plt.title("Curvas ROC de los Modelos")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.legend(loc='lower right')
plt.show()
