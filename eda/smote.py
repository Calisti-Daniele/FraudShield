import pandas as pd
from imblearn.over_sampling import SMOTE

# Carica il dataset originale
data = pd.read_csv('../datasets/merged_creditcard.csv')

# Separazione tra feature (X) e target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Riduci il numero di campioni di classe 1 con SMOTE (configura il rapporto desiderato)
# ratio = 0.5 significa che vogliamo metà dei campioni della classe 0
smote = SMOTE(sampling_strategy=0.8, random_state=42)  # Configura il rapporto di bilanciamento
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combina i dati riequilibrati
balanced_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)

# Salva il nuovo dataset
balanced_data.to_csv('../datasets/balanced_creditcard.csv', index=False)

# Stampa le informazioni del nuovo dataset
print(f"Nuovo dataset salvato con distribuzione delle classi:\n{balanced_data['Class'].value_counts()}")
