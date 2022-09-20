from cProfile import label
import pandas as pd

df = pd.read_csv("trainSet.csv")

Y = df["label"].values

from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)

X = df.drop(labels=["label"], axis=1)

# from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC(max_iter= 40000)
model = RandomForestClassifier(n_estimators = 20, random_state = 42)
model.fit(X,Y)

df_test = pd.read_csv("testSet.csv")

y_test = df_test["label"].values
y_test = LabelEncoder().fit_transform(y_test)

x_test = df_test.drop(labels=["label"], axis=1)

prediction = model.predict(x_test)

from sklearn import metrics

print("Accuracy",metrics.accuracy_score(y_test,prediction))