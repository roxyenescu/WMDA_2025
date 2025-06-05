### **Exercise 5: Applying a Classification Model**
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Incarcarea setului de date pentru Titanic
df = sns.load_dataset("titanic")

# Step 2: Ingineria pentru caracteristici (from Exercise 4)
df["family_size"] = df["sibsp"] + df["parch"] + 1  # Create new feature
df["age"].fillna(df["age"].median(), inplace=True)  # Handle missing age values
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)  # Handle missing embarked values
df.dropna(subset=["fare"], inplace=True)  # Remove rows with missing fare values
df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)  # One-hot encoding

# Selectarea caracteristicilor si scalarea numerica pe coloane
features = ["age", "fare", "family_size", "sex_male", "embarked_Q", "embarked_S"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Definirea variabilel Target si impartirea datelor in set de antrenare si test (80% - 20%)
X = df[features]
y = df["survived"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea modelului de Regresie Logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictii
y_pred = model.predict(X_test)

# Evaluarea performantei modelului prin scorul de acuratete
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
