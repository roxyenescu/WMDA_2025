# Exercise 1 (15 minutes): Testing Perceptron Efficiency on Different Data Types
import numpy as np
from sklearn.metrics import accuracy_score

# Assumed to be provided in course materials
class Perceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # bias trick
        self.weights = np.random.randn(X.shape[1])
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = self.predict_single(xi)
                error = yi - pred
                self.weights += self.lr * error * xi

    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) >= 0 else 0

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.array([self.predict_single(xi) for xi in X])

# 1. AND gate dataset (linearly separable)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

model_and = Perceptron()
model_and.fit(X_and, y_and)
pred_and = model_and.predict(X_and)

print("AND gate accuracy:", accuracy_score(y_and, pred_and))

# 2. XOR gate dataset (non-linearly separable)
X_xor = X_and
y_xor = np.array([0, 1, 1, 0])

model_xor = Perceptron()
model_xor.fit(X_xor, y_xor)
pred_xor = model_xor.predict(X_xor)

print("XOR gate accuracy:", accuracy_score(y_xor, pred_xor))
