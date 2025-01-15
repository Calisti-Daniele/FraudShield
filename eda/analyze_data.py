import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carico i dati
creditcard_data = pd.read_csv('../datasets/ready_to_use/selected_features_creditcard.csv')

# Ispeziono le prime righe
print(creditcard_data.head())

# Controllo informazioni sul dataset
print(creditcard_data.info())

# Distribuzione delle classi
class_counts = creditcard_data['Class'].value_counts()
print(f"Distribuzione delle classi:\n{class_counts}")

# Visualizzo la distribuzione delle classi
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Distribuzione delle Classi')
plt.xlabel('Classe (0 = Normale, 1 = Frode)')
plt.ylabel('Frequenza')
plt.show()


# Visualizzo correlazioni tra feature
plt.figure(figsize=(12, 10))
correlation_matrix = creditcard_data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Matrice di Correlazione')
plt.show()
