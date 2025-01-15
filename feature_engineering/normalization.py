from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carica il dataset
data = pd.read_csv('../datasets/balanced_creditcard.csv')

# Identifica le colonne numeriche (escludendo la target 'Class')
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.drop('Class')

# Applica la normalizzazione alle colonne numeriche
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Stampa un riepilogo delle statistiche per verificare la normalizzazione
print(data[numerical_columns].describe())

# Salva il dataset aggiornato
data.to_csv('../datasets/ready_to_use/normalized_creditcard.csv', index=False)
print("Dataset normalizzato e salvato.")
