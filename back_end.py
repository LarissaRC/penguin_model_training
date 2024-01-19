# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
# For plotting graphs
from matplotlib import pyplot as plt
# Import the sklearn library for KNN
from sklearn.neighbors import KNeighborsClassifier
# Import joblib for saving and loading models
import joblib

# Import the csv file
df = pd.read_csv('penguins_size.csv')

# Delete lines with NaN
df = df.dropna()

print (df.head())

# Prepare the training set
X = df.loc[:,'culmen_length_mm':'flipper_length_mm']
Y = df.loc[:,'species']

X.describe()

from sklearn.model_selection import train_test_split
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=3)

knn = KNeighborsClassifier()
# Train the model
knn.fit(X, Y)

knn_test = KNeighborsClassifier()
# Train the model
knn_test.fit(X_treino, Y_treino)

# Save the trained model to a file
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(knn_test, 'knn_test_model.pkl')

# Reload the trained model from the file
loaded_model_knn = joblib.load('knn_test_model.pkl')

from sklearn import preprocessing, metrics
from sklearn.metrics import classification_report, confusion_matrix

# Applying the predictions using the test data created before
Y_predicoes = loaded_model_knn.predict(X_teste)

# MODEL EVALUATION
# LET'S EVALUATE THE REAL VALUE OF THE Y_TESTE DATASET WITH THE PREDICTIONS
print("ACURÁCIA DO MODELO KNN: ", metrics.accuracy_score(Y_teste, Y_predicoes))
print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(Y_teste, Y_predicoes))
print("MATRIZ DE CONFUSÃO:")
print(confusion_matrix(Y_teste, Y_predicoes))