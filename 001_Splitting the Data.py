from sklearn import datasets
sugar = datasets.load_diabetes()

x = sugar.data
y = sugar.target

import pandas as pd
frame = pd.DataFrame(x)

frame.columns = sugar.feature_names

#----------------------------------------------------------------------

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.3, random_state=69)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
algo1 = LinearRegression()
algo1.fit(x_train, y_train)
y_pred = algo1.predict(x_test)