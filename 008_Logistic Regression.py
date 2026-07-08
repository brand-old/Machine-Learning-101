from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.3, random_state=42)


class_ = LogisticRegression( C = 2, max_iter = 1000, solver = 'liblinear')
class_.fit(x_train, y_train)

y_pred = class_.predict(x_test)

score_train = class_.score(x_train, y_train)
score_test = class_.score(x_test, y_test)

#class_.predict(cancer.data) - cancer.target
#class_.predict_proba(cancer.data)

print(round(score_train, 2))
print(f"{score_test:.2f}")