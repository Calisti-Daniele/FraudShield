from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Carica il dataset
data = pd.read_csv('../datasets/ready_to_use/normalized_creditcard.csv')

# Separazione tra feature e target
X = data.drop('Class', axis=1)  # Rimuovo la colonna target
y = data['Class']  # Target

# Applica SelectKBest con la funzione f_classif
selector = SelectKBest(score_func=f_classif, k=16)  # Seleziona le 10 migliori feature
X_new = selector.fit_transform(X, y)

# Ottieni i nomi delle feature selezionate
selected_features = X.columns[selector.get_support()]

# Visualizza le feature selezionate
print("Feature selezionate:")
print(selected_features)

# Salva il nuovo dataset con le feature selezionate
selected_data = pd.DataFrame(X_new, columns=selected_features)
selected_data['Class'] = y  # Riaggiungo la colonna target
selected_data.to_csv('../datasets/ready_to_use/selected_features_creditcard.csv', index=False)

print("Dataset con feature selezionate salvato come 'selected_features_creditcard.csv'.")
