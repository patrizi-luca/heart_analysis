import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Carica il tuo dataset (assicurati che sia già pulito e preprocessato)
# Ad esempio, se il tuo DataFrame è chiamato df:
# df = pd.read_csv('tuo_file.csv')
df = pd.read_csv('heart.csv')

# Crea un oggetto LabelEncoder
label_encoder = LabelEncoder()

# Applica il label encoding alla colonna 'Sex' nel DataFrame
df['Sex']               = label_encoder.fit_transform(df['Sex'])
df['ChestPainType']     = label_encoder.fit_transform(df['ChestPainType'])
df['RestingECG']        = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina']    = label_encoder.fit_transform(df['ExerciseAngina'])
df['ExerciseAngina']    = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope']          = label_encoder.fit_transform(df['ST_Slope'])



# Definisci le feature (X) e il target (y)
X = df.drop('HeartDisease', axis=1) # Separo la colonna di outcome per il training
y = df['HeartDisease']

# Suddividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializza il modello Random Forest
model = RandomForestClassifier(random_state=42)

# Addestra il modello
model.fit(X_train, y_train)

# Effettua previsioni sul test set
y_pred = model.predict(X_test)

# Valuta le prestazioni del modello
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
