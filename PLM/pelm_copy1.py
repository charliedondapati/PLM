import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

from BELM.belm import BELM


class PLM(ELM):
    def __init__(self, inputs=0, outputs=0, classification="", w=None, batch=1000, accelerator=None,
                 precision='double', norm=None, tprint=5):
        super().__init__(inputs, outputs, classification, w, batch, accelerator, precision, norm, tprint)

    def plm_train(self, data, target, label, n, s1, s2, c, acc=None):
        """ Progressive learning implementation"""

        gamma = 0.01 + 1 * 0.005
        nnet4 = []
        v4 = []
        yhat4 = []
        index4 = []
        var = s2
        train_data = []
        train_target = []
        train_label = []
        real_train_label = []

        for n_c in range(0, c):
            # yxf
            num_node = []
            error = []
            nn_optimal = []
            p_max = -1
            s2 = var

            for nn in range(0, n):
                # wsn

                for n_s1 in range(0, s1):

                    if nn == 0:
                        index = np.random.permutation(data.shape[0])
                        X_test = data[index]
                        Y_test = target[index]
                        L_test = label[index]
                        X_train = X_test[:5, :]
                        Y_train = Y_test[:5, :]

                    for n_s2 in range(0, s2):
                        belm = BELM(X_train.shape[1], Y_train.shape[1], precision="single")
                        belm.add_neurons(5, 'sigm')
                        belm.train(X_train[:5, :], Y_train[:5, :])
                        yhat = belm.predict(X_test)

                        v = np.abs(Y_test - yhat)
                        v = np.where(v > gamma, 0, v)
                        v = np.where(v > 0, 1, v)

                        num_node.append(np.sum(v))
                        error.append(belm.error(Y_test, yhat))
                        # print(num_node)
                        if max(num_node) > p_max:
                            p_max = max(num_node)
                            e1 = error[num_node.index(max(num_node))]
                            nnet1 = belm
                            v1 = v
                            # yhat1 = yhat
                            index1 = index

                # data1=[y phi]
                # data = []
                nn_optimal.append((max(num_node), error[num_node.index(max(num_node))]))
                Y_test = target[index1]
                X_test = data[index1]
                L_test = label[index1]
                new_ind = np.where(v1 == 1)[0]
                Y_train = Y_test[new_ind]
                X_train = X_test[new_ind]
                L_train = L_test[new_ind]

                s2 = 1

            nnet4.append(nnet1)
            if len(train_data) == 0:
                train_data = X_train
                train_target = Y_train
                real_train_label = L_train
                train_label = np.full_like(L_train, n_c + 1)

            else:
                train_data = np.vstack((train_data, X_train))
                train_target = np.vstack((train_target, Y_train))
                real_train_label = np.vstack((real_train_label, L_train))
                train_label = np.vstack((train_label, np.full_like(L_train, n_c + 1)))

            # removing  data points of the first cluster
            # only data points where the labels are wrongly identified are selected
            new_ind = np.where(v1 == 0)[0]
            data = data[new_ind]
            target = target[new_ind]
            label = label[new_ind]

        return train_data, train_target, train_label, real_train_label, nnet4

    def plm_test(self, train_dat, train_lab, test_dat, test_tar, test_lab, nn, c):
        # SVM classifier
        clf = svm.SVC()
        clf.fit(train_dat, train_lab.ravel())
        predicted = clf.predict(test_dat)
        svm_acc = metrics.accuracy_score(test_lab, predicted)
        # print("SVM Accuracy: ", metrics.accuracy_score(test_lab, predicted))

        # error = []
        final_tar = []
        final_pred = []
        for n_c in range(0, c):
            r_ind = np.where(test_lab == n_c + 1)[0]
            # p_ind = np.where(predicted == n_c + 1)[0]
            tmp_dat = test_dat[r_ind]
            tmp_tar = test_tar[r_ind]
            # tmp_lab = test_lab[ind]
            test_pred = nn[n_c].predict(tmp_dat)
            # error.append(nn[n_c].error(tmp_tar, test_pred))
            if n_c == 0:
                final_tar = tmp_tar
                final_pred = test_pred
            else:
                final_tar = np.vstack((final_tar, tmp_tar))
                final_pred = np.vstack((final_pred, test_pred))

        return np.mean((final_pred - final_tar) ** 2), svm_acc
