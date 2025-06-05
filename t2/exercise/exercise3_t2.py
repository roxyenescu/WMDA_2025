import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Incarcarea setului de date
wine = load_wine()
X = wine.data
y = wine.target

# Convertirea setului de date intr-un DataFrame
df = pd.DataFrame(X, columns=wine.feature_names)
df["target"] = y
print(df)

# Impartirea setului de date in set de antrenare si de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea clasificatorului Decision Tree
model_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Verificarea acuratetii pe setul de test
accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy:.2f}")

# Vizualizarea structurii
plt.figure(figsize=(10, 6))
plot_tree(
    model_dt,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True
)
plt.show()

# Importanta caracteristicilor
importances = model_dt.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': wine.feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance_df)