import numpy as np
import pandas as pd

from PLM.pelm import pelm

if __name__ == '__main__':
    df = pd.read_csv('equ23_data.csv', header=None)
    target = np.array(df[df.columns[[-2, -1]]])
    data = np.array(df.iloc[:, :-2])

    n = 10;
    parameter1 = 10;
    parameter2 = 10;
    model_number = 3;

    pelm(data, target, model_number, n=n, p=parameter1, s=parameter2)
