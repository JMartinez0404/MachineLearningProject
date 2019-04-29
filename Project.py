import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#df = pd.read_excel(r"C:\Users\Joel\Desktop\MovieData.xlsx")
df = pd.read_excel(r"MovieData.xlsx")

df = pd.DataFrame(df,columns=['Gross USA','Opening Weekend USA','Worldwide Gross','Budget','Metascore','IMDb Rating','Tomatometer','Audience Score'])

X = df[['Gross USA','Opening Weekend USA','Worldwide Gross','Budget','Metascore','IMDb Rating','Tomatometer']]
y = df['Audience Score']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)
predictions = regr.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


# search for an optimal value of K for KNN
k_range = list(range(1, 15))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
# Most optimal is with a k = 6

knn = KNeighborsClassifier(n_neighbors=6)
print(cross_val_score(knn, X, y, cv=3, scoring='accuracy').mean())

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean())
# This shows that Knn would not be a good approach for this model but is more optimal than a logistic regression

