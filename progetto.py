import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Caricamento del dataset
file_path = "C:/Users/anton/Desktop/Electric_Vehicle_Population_Data.csv"
df = pd.read_csv(file_path)

# Visualizzazione delle informazioni generali del dataset
df.info()
print(df.describe())
print("Dataset shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())

# Rimozione colonne non necessarie
df = df.drop(columns=["VIN (1-10)", "Model"], errors='ignore')

# Rimozione dei valori mancanti
df = df.dropna()

# Rimuoviamo i veicoli con autonomia pari a 0
df = df[df["Electric Range"] > 0]

# Codifica delle variabili categoriche
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Definizione delle variabili indipendenti (X) e della variabile target (y)
X = df.drop(columns=["Electric Range"], errors='ignore')
y = df["Electric Range"]

# Controllo dei valori minimi e massimi della variabile target
y_min, y_max = y.min(), y.max()
print(f"Valore minimo di Electric Range: {y_min}, Valore massimo: {y_max}")
print("Distribuzione dei valori target:")
print(y.value_counts().head(10))

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Suddivisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definizione dei modelli
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Addestramento e valutazione dei modelli
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    print(f"\n{name} Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Training Time:", round(end_time - start_time, 2), "seconds")

    # Stampa dettagliata delle prime 5 predizioni
    print("Esempio di 5 predizioni:")
    for i in range(5):
        print(f"Veicolo {i + 1}: Previsto = {y_pred[i]:.2f}, Reale = {y_test.iloc[i]:.2f}")

# Analisi dell'importanza delle feature
rf = models["Random Forest"]
feature_importance = rf.feature_importances_
features = df.drop(columns=["Electric Range"], errors='ignore').columns
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], feature_importance[sorted_idx], color='blue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.show()

# Visualizzazione dell'albero di decisione
dt = models["Decision Tree"]
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=features, filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree Visualization")
plt.show()
