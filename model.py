import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset

iris_df = pd.read_csv('data/iris.csv')

# selecting features and target data

X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# Split data into train test sets
# 70% draining and 30% test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# Create an instance of the Kneighbor classifier
clf = KNeighborsClassifier(n_neighbors=10)

# train the classifer on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy

accuracy = accuracy_score(y_test, y_pred)
print(f"Acuracy: {accuracy}")  # Accuracy : 0.91


# save model to the disk
joblib.dump(clf, "output_models/kn_model.sav")
