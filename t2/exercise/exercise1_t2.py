import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Incarcarea setului de date
wine = load_wine()

X = wine.data
y = wine.target

# Convertirea setului de date intr-un DataFrame
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
print(df.head())

# Impartirea setului de date in set de antrenare si de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea clasificatorului Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predictia pe setul de test
y_pred = model.predict(X_test)

# Afisarea scorului de acuratete
accuracy = accuracy_score(y_test, y_pred)
print(f"Scorul de acuratete: {accuracy:.2f}")

# Matricea de confuzie
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)



