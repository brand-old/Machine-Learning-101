import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('FuelEconomy.csv', delimiter=',')
X = data[:, 0]
y = data[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

###########################
##### Start from here #####
###########################

def fit(x_train, y_train):
    ## Write code for fit function
    m = ((x_train*y_train).mean() - (x_train.mean()*y_train.mean())) / ((x_train**2).mean() - (x_train.mean())**2)
    c = y_train.mean() - (m*x_train.mean())
    return(m,c)
    pass


def predict(x, m, c):
    ## Write code for predict function
    y_pred = (m*x) + c
    return(y_pred)
    pass

def score(y_truth, y_pred):
    ## Write code for score function
    runs = 1 - ((((y_truth - y_pred)**2).mean()) / (((y_truth - y_pred.mean())**2).mean()))
    return(runs)
    pass


def cost(x, y, m, c):
    ## Write code for cost function
    CD = ((y-(m*x+c))**2).mean()
    return(CD)
    pass


###########################
###########################

m, c = fit(X_train, Y_train)

y_test_pred = predict(X_test, m, c)
y_train_pred = predict(X_train, m, c)

print(round(score(Y_test, y_test_pred), 2))
print(round(score(Y_train, y_train_pred), 2))
print(round(cost(X_train,Y_train, m, c ), 2))