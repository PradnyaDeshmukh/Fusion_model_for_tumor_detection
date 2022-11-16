from cProfile import label
import pandas as pd

df3 = pd.read_csv("trainRGB.csv")

# df3.columns.str.match("Unnamed")
# df3.loc[:,~df3.columns.str.match("Unnamed")]
# print(df3.head())
Y = df3["label"].values

from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)

X = df3.drop(labels=["label"], axis=1)

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC(max_iter= 40000)
model = RandomForestClassifier()
model.fit(X,Y)

# rfc = RandomForestClassifier()
# parameters = {
#     "n_estimators":[1,2,3,4,5],
#     "max_depth":[1,2,3,4,8,16,32,None]
    
# }

# def display(results):
#     print(f'Best parameters are: {results.best_params_}')
#     print("\n")
#     mean_score = results.cv_results_['mean_test_score']
#     std_score = results.cv_results_['std_test_score']
#     params = results.cv_results_['params']
#     for mean,std,params in zip(mean_score,std_score,params):
#         print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

# from sklearn.model_selection import GridSearchCV
# cv = GridSearchCV(rfc,parameters,cv=5)
# cv.fit(X,Y)
# display(cv)









df_test = pd.read_csv("testRGB.csv")
# print(df_test.head())

# df_test.columns.str.match("Unnamed")
# df_test.loc[:,~df_test.columns.str.match("Unnamed")]
y_test = df_test["label"].values
y_test = LabelEncoder().fit_transform(y_test)

x_test = df_test.drop(labels=["label"], axis=1)

prediction = model.predict(x_test)
# prediction = cv.predict(x_test)

from sklearn import metrics

print("Accuracy",metrics.accuracy_score(y_test,prediction))
