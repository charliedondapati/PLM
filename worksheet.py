import hpelm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('BELM/sample_data.csv', header=None)
output = np.array(df[df.columns[-1]])
data = np.array(df.iloc[:, :-1])

X_train, X_test, Y_train, Y_test = train_test_split(data, output, test_size=0.3)
Y_train = Y_train.reshape(-1, 1)

elm_gpu = hpelm.ELM(X_train.shape[1], Y_train.shape[1], precision="single", accelerator="GPU")
elm_gpu.add_neurons(10, 'sigm')
elm_gpu.train(X_train, Y_train)
P = elm_gpu.predict(X_test)
e = elm_gpu.error(Y_test, P)
print("Error:  ", e)
