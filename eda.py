import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('heart.csv')
print(df.head())
# Ottieni statistiche descrittive per tutte le colonne
print(df.describe())

# Ottieni informazioni sulle colonne, inclusi i tipi di dati e i valori non nulli
print(df.info())

sns.set_theme(style='whitegrid')  # Imposta uno sfondo con griglia
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='blue')
plt.title('Distribuzione dell\'età')
plt.xlabel('Età')
plt.ylabel('Frequenza')
plt.show()

# Matrice di correlazione
correlation_matrix = df.corr()

# Visualizzazione della matrice di correlazione con heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Matrice di Correlazione')
plt.show()


# Conteggio dei valori mancanti per ogni colonna
print("Valori Mancanti",df.isnull().sum())

# Conteggio dei valori per una variabile categorica (es. 'Sex')
sns.countplot(x='Sex', data=df)
plt.title('Distribuzione tra i sessi')
plt.show()
