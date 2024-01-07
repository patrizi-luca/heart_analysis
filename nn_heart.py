import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Carica il tuo dataset
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
X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values

# Suddividi il dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizza le feature (è importante normalizzare per le reti neurali)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converti i dati in Tensori di PyTorch
X_train_tensor  = torch.FloatTensor(X_train)
y_train_tensor  = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor   = torch.FloatTensor(X_test)
y_test_tensor   = torch.FloatTensor(y_test).view(-1, 1)

# Crea un DataLoader per gestire i dati
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Definisci il modello di rete neurale
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Inizializza il modello
model = NeuralNetwork()

# Definisci la funzione di perdita e l'ottimizzatore
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Addestra il modello
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Effettua previsioni sul test set
with torch.no_grad():
    y_pred_tensor = (model(X_test_tensor) > 0.5).float()

# Converte i risultati in array NumPy
y_pred_nn = y_pred_tensor.numpy().astype(int)
y_test_np = y_test_tensor.numpy().astype(int)

# Valuta le prestazioni del modello
accuracy_nn = accuracy_score(y_test_np, y_pred_nn)
conf_matrix_nn = confusion_matrix(y_test_np, y_pred_nn)
classification_rep_nn = classification_report(y_test_np, y_pred_nn)

print(f'Accuracy Neural Network (PyTorch): {accuracy_nn:.4f}')
print(f'Confusion Matrix Neural Network (PyTorch):\n{conf_matrix_nn}')
print(f'Classification Report Neural Network (PyTorch):\n{classification_rep_nn}')
