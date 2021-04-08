from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from time import time

from BELM.belm import BELM


def cpu_belm(trainX, testX, trainY, testY):
    belm = BELM(trainX.shape[1], trainY.shape[1], precision="single")
    belm.add_neurons(2, 'sigm')
    st_time = time()
    belm.train(trainX, trainY)

    yhat = belm.predict(testX)
    e = belm.error(testY, yhat)

    print("The error:", e)
    print("Execution time:", time() - st_time)


def gpu_belm(trainX, testX, trainY, testY):
    belm = BELM(trainX.shape[1], trainY.shape[1], precision="single", accelerator="GPU")
    belm.add_neurons(4, 'sigm')
    st_time = time()
    belm.train(trainX, trainY)

    yhat = belm.predict(testX)
    e = belm.error(testY, yhat)

    print("The error:", e)
    print("Execution time:", time() - st_time)


if __name__ == '__main__':
    df = pd.read_csv('sample_data.csv', header=None)
    output = np.array(df[df.columns[-1]])
    data = np.array(df.iloc[:, :-1])

    X_train, X_test, Y_train, Y_test = train_test_split(data, output, test_size=0.3)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    cpu_belm(X_train, X_test, Y_train, Y_test)
    # gpu_belm(X_train, X_test, Y_train, Y_test)
