import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


dates = []
prices = []

def get_data (filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return


def predict_prices (dates, prices, x):  
    dates = np.reshape(dates, (len(dates), 1))
    svr_len = SVR(kernel = 'linear', C=1e3)
    svr_poly = SVR(kernel ='poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel ='rbf', C=1e3, gamma = 0.1)
    svr_len.fit (dates, prices)
    svr_poly.fit (dates, prices)
    svr_rbf.fit (dates, prices)
    
    plt.scatter (dates, prices, color = 'black', label= 'data')
    plt.plot(dates, svr_rbf.predict(dates), color = 'red', label = 'RBF model')
    plt.plot(dates, svr_len.predict(dates), color = 'green', label = 'linear model')
    plt.plot(dates, svr_poly.predict(dates), color = 'blue', label = 'poly model')
    plt.xlabel('dates')
    plt.ylabel('prices')
    plt.title('support vector regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_len.predict(x)[0], svr_poly.predict(x)[0]


get_data('TSLA.csv')
predicted_price = predict_prices(dates, prices, 29)
print (predicted_price) 