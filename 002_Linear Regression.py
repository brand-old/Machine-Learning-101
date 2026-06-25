import numpy as np
data = np.loadtxt("FuelEconomy.csv", delimiter=",")
#print(data.shape)   # =  (100,2) #
x = data[:,0].reshape(-1,1)
y=data[:,1]

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.3, random_state = 42)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
algo1 = LinearRegression()
algo1.fit(x_train, y_train)

print(np.round(algo1.coef_[0], 2))
print(np.round(algo1.intercept_, 2))