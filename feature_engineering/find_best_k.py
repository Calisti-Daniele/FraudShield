from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Carica il dataset
data = pd.read_csv('../datasets/ready_to_use/normalized_creditcard.csv')

# Separazione tra feature e target
X = data.drop('Class', axis=1)
y = data['Class']

# Prova diversi valori di k
k_values = range(1, X.shape[1] + 1)  # Da 1 al numero totale di feature
best_k = 0
best_score = 0
scores = []

for k in k_values:
    # Applica SelectKBest con k feature
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Addestra un modello di Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42, verbose=True)

    # Valuta il modello con cross-validation
    cv_score = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc').mean()
    scores.append(cv_score)

    # Aggiorna il miglior k se necessario
    if cv_score > best_score:
        best_score = cv_score
        best_k = k

# Stampa il miglior valore di k e il punteggio associato
print(f"Miglior valore di k: {best_k}")
print(f"Punteggio ROC-AUC: {best_score}")

# Grafico delle prestazioni in funzione di k
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker='o')
plt.title('Prestazioni in funzione di k')
plt.xlabel('Numero di feature selezionate (k)')
plt.ylabel('ROC-AUC')
plt.grid()
plt.show()
