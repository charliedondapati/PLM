from hpelm.nnets import SLFN
import numpy as np


class BSLFN(SLFN):
    """ """

    def __init__(self, inputs, outputs, norm=None, precision=np.float64):
        super().__init__(inputs, outputs, norm, precision)
        self.H = None

        self.func["rsigm"] = lambda X: - np.log((1 / X) - 1)
        self.func["ssigm"] = lambda X, W, B: 1 / (1 + np.exp(np.dot(X, W) - B))

    def add_neurons(self, number, func, W, B):
        # print("in bslfn")
        ntypes = [nr[1] for nr in self.neurons]  # existing types of neurons
        if func in ntypes:
            # add to an existing neuron type
            i = ntypes.index(func)
            nn0, _ = self.neurons[i]
            number = nn0 + number
            self.neurons[i] = (number, func)
        else:
            # create a new neuron type
            # print("BSNLF:",self.neurons)
            self.neurons.append((number, func))
            # print(self.neurons)
        self.reset()
        self.B = None

    def _delta(self, X, a, b, T):
        """Predict a batch of data. Auxiliary function that implements a particular prediction.
        For prediction, use `ELM.predict()` instead.
        Args:
            X (matrix): input data size (N * `inputs`)
        Returns:
            Y (matrix): predicted outputs size (N * `outputs`), always in float/double format.
        """
        # assert self.B is not None, "Solve the task before predicting"
        H = self.func["sigm"](X, a, b)
        # H = self._project(X)
        self.B = self.solve_corr(H, T)
        Y = np.dot(H, self.B)
        return np.array(T - Y)

    def update_weights(self, H, X, delta):

        tempH = self.func["rsigm"](H)
        X_inv = np.linalg.pinv(X)
        a = np.dot(X_inv, tempH)
        b = np.sqrt(np.mean((tempH - np.dot(X, a)) ** 2))
        c = self.func["ssigm"](X, a, b)
        c = np.dot(X_inv, c - tempH)
        newH = self.func['ssigm'](X, a - c, b)
        # newH = self.func['ssigm'](X, a, b)

        # newH = self.func['sigm'](X, a, b)
        newH = self.H_scaler.inverse_transform(newH)
        newB = np.dot(np.linalg.pinv(newH), delta)

        return a, b, newB

    def solve_bslfn(self, X, T):
        # nn, _, _, _ = self.neurons["sigm"]

        weights = np.zeros((self.inputs, 1))
        bias = np.zeros(1)
        beta = np.zeros((1, 1))
        count = 1
        while count <= self.L:
            # print("\n\nsolve_bslfn:count=",count)
            # generate input weights and bias
            a0 = np.random.randn(self.inputs, 1)
            a0 *= 3.0 / self.inputs ** 0.5

            b0 = np.random.randn(1)
            # b0 = np.abs(b0)
            # b0 *= self.inputs

            delta = self._delta(X, a0, b0, T)

            # return

            from sklearn.preprocessing import MinMaxScaler
            self.H_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            B_pinv = np.linalg.pinv(self.B)
            # hh=np.dot(delta, B_pinv)
            tempH = self.H_scaler.fit_transform(np.dot(delta, B_pinv))

            a, b, B = self.update_weights(tempH, X, delta)

            # save weights, bias , H and beta(output weights)
            weights = np.hstack((weights, a))
            bias = np.hstack((bias, b))
            beta = np.vstack((beta, B))

            count += 1

        self.neurons = [(self.L, "sigm", weights[:, 1:], bias[1:])]
        self.B = beta[1:, :]

    def _predict_belm(self, X):
        """Predict a batch of data. Auxiliary function that implements a particular prediction.
        For prediction, use `ELM.predict()` instead.
        Args:
            X (matrix): input data size (N * `inputs`)
        Returns:
            Y (matrix): predicted outputs size (N * `outputs`), always in float/double format.
        """
        assert self.B is not None, "Solve the task before predicting"
        H = self._project(X)
        H = self.H_scaler.inverse_transform(H)
        Y = np.dot(H, self.B)

        return Y
