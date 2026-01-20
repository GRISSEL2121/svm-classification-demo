import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Dataset sintético
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_redundant=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Construcción del Pipeline:
# Encapsulamos el escalado y el modelo en un solo flujo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Paso de preprocesamiento
    ('model', SVC(kernel='rbf', C=1.0, gamma='scale'))  # Estimador
])

# Entrenamiento:
# El scaler solo aprende de X_train y luego transforma X_train 
# antes de pasarlo al modelo SVC.
pipeline.fit(X_train, y_train)

# Al llamar a predict, X_test se escala automáticamente usando 
# los parámetros aprendidos de X_train (evitando data leakage).
y_pred = pipeline.predict(X_test)

# Resultados
print("--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))

print("--- Matriz de Confusión ---")
print(confusion_matrix(y_test, y_pred))

# Ejemplo de uso con un nuevo dato
nuevo_dato = np.random.randn(1, 10) 
prediccion = pipeline.predict(nuevo_dato)
print(f"\nPredicción para el nuevo dato: {prediccion[0]}")