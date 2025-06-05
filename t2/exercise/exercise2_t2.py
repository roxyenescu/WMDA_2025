import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# incarcarea setului de date
wine = load_wine()
X = wine.data
y = wine.target

# Convertirea intr-un DataFrame
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
print(df)

# Impartirea setului de date in set de antrenare si de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# Antrenarea clasificatorului Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

# Andrenarea modelului de regresie logistica
model_rl = LogisticRegression(max_iter=2000)
model_rl.fit(X_train, y_train)
y_pred_lg = model_rl.predict(X_test)

# Compararea metricilor: acuratete, precizie, recall
metrics = {}
for model_name, y_pred in [("Naive Bayes", y_pred_nb), ("Logistic Regression", y_pred_lg)]:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

# Afisarea rezultatelor
for model_name, scores in metrics.items():
    print(f"=== {model_name} ===")
    print(f"Accuracy: {scores['Accuracy']:.2f}")
    print(f"Precision: {scores['Precision']:.2f}")
    print(f"Recall: {scores['Recall']:.2f}")
    print()

# Matricea de confuzie pentru fiecare model
from sklearn.metrics import confusion_matrix
print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lg))