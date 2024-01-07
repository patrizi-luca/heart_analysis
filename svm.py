import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Suddividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializza il modello SVM
model_svm = SVC(random_state=42)

# Addestra il modello
model_svm.fit(X_train, y_train)

# Effettua previsioni sul test set
y_pred_svm = model_svm.predict(X_test)

# Valuta le prestazioni del modello SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)

print(f'Accuracy SVM: {accuracy_svm:.4f}')
print(f'Confusion Matrix SVM:\n{conf_matrix_svm}')
print(f'Classification Report SVM:\n{classification_rep_svm}')
