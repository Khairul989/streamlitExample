import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.header("By: Kerol")
st.title("Auto prediction for KNN, SVM, Random Forest")

st.write("""
Selecting the best classifier
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer","Wine Dataset"))

cl_name = st.sidebar.selectbox("Select Classifier", ("KNN","Svm","Random Forrest"))

def get_class(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_class(dataset_name)

st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

def get_param(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1,15)
        params["K"] = K
    elif clf_name == "Svm":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimator = st.sidebar.slider("n_estimator", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimator
    return params

params = get_param(cl_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Svm":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimator"], max_depth=params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(cl_name, params)

##Making prediction
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {cl_name}")
st.write(f"Accuracy = {acc}")

##plotting
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

st.pyplot()