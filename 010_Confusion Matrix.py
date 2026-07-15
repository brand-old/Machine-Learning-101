from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

eye = datasets.load_iris()

x_tr, x_te, y_tr, y_te = train_test_split(eye.data, eye.target, random_state = 42)

#print(eye)
#print("EYE+TARGET", eye.target)
#print("Shape of y_tr:", y_tr)
#print("Shape of x_tr:", x_tr)

clf = LogisticRegression()
clf.fit(x_tr, y_tr)

ytr_pred = clf.predict(x_tr)
yte_pred = clf.predict(x_te)

#print(confusion_matrix(y_tr, ytr_pred))
print(confusion_matrix(y_te, yte_pred))