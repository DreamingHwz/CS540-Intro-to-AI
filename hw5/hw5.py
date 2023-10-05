import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data(filepath):
    csvlist = list()
    with open(filepath, 'r', encoding='UTF-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row['year'])
            days = int(row['days'])
            csvlist.append([year, days])
    csvlist = np.array(csvlist)

    return csvlist

def list_plt(clist):
    x = clist[:, 0]
    y = clist[:, 1]
    plt.plot(x, y)
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')
    plt.savefig("plot.jpg")

    return

def Q3a(clist):
    X = list()
    for x in clist[:, 0]:
        X.append([1, x])
    X = np.array(X)

    return X

def Q3b(clist):
    Y = clist[:, 1]
    Y = np.array(Y)

    return Y

def Q3c(X):
    X_T = np.transpose(X)
    Z = np.dot(X_T, X)

    return Z

def Q3d(Z):
    I = np.linalg.inv(Z)

    return I

def Q3e(Z, X):
    X_T = np.transpose(X)
    PI = np.dot(Z, X_T)

    return PI

def Q3f(PI, Y):
    beta_hat = np.dot(PI, Y)

    return beta_hat

def Q4(x_test, hat_beta):
    y_test = hat_beta[0] + hat_beta[1] * x_test

    return y_test

def Q5a(hat_beta):
    if (hat_beta[1] > 0):
        return '>'
    elif (hat_beta[1] < 0):
        return '<'
    else:
        return '='

def Q6a(y, hat_beta):
    x = (y - hat_beta[0])/hat_beta[1]

    return x


if __name__ =="__main__":
    filename = sys.argv[1]
    clist = load_data(filename)
    list_plt(clist)

    X = Q3a(clist)
    print("Q3a:")
    print(X)

    Y = Q3b(clist)
    print("Q3b:")
    print(Y)

    Z = Q3c(X)
    print("Q3c:")
    print(Z)

    I = Q3d(Z)
    print("Q3d:")
    print(I)

    PI = Q3e(I, X)
    print("Q3e:")
    print(PI)

    hat_beta = Q3f(PI, Y)
    print("Q3f:")
    print(hat_beta)

    y_test = Q4(2021, hat_beta)
    print("Q4: " + str(y_test))

    print("Q5a: " + Q5a(hat_beta))

    print("Q5b: " + "If the sign of beta_1 is >, that means the amount of freezing days \
of the lake each year is generally increasing from 1855 to 2020, which is \
concluded by our linear regression model. If it is <, that means the \
amount of freezing days each year is generally decreasing. If it is =, that means \
the amount fluctuate but is generally similar between years.")

    print("Q6a: ", end="")
    print(Q6a(0, hat_beta))

    print("Q6b: " + "If there are few data, like in toy.csv, I don't think it is \
a compelling prediction because the data set is too small to predict a trend, \
and the freezing days fluctuates every year. If there are enough data, like in \
hw5.csv, I think it is compelling, because we can see from the plot that the days \
did decrease by year in general, and maybe the lake will no longer freeze one day, \
if the change is linear. Also maybe we can predict x* with non-linear model next time, \
to make it more realistic.")