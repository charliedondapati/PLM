import numpy as np
from hpelm.nnets.slfn_skcuda import SLFNSkCUDA
from numpy import dtype
from pycuda import gpuarray


class BSLFNskCUDA(SLFNSkCUDA):
    """ """

    def __init__(self, inputs, outputs, norm=None, precision=np.float64):
        super().__init__(inputs, outputs, norm, precision)
        self.H = None

        self.func["rsigm"] = lambda X: - np.log((1 / X) - 1)

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
            self.neurons.append((number, func))
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
        devX = gpuarray.to_gpu(X)
        deva = gpuarray.to_gpu(a)
        devb = gpuarray.to_gpu(b)
        H = self.func["sigm"](devX, deva, devb).get()
        # H = self._project(X)
        self.B = self.solve_corr(H, T)
        Y = np.dot(H, self.B)
        return np.array(T - Y)

    def solve_corr(self, HH, HT):
        HH_pinv = np.linalg.pinv(HH)
        B = np.dot(HH_pinv, HT)
        return B

    def update_weights(self, H, X, delta):

        tempH = self.func["rsigm"](H)
        X_inv = np.linalg.pinv(X)
        a = np.dot(X_inv, tempH)
        b = np.sqrt(np.mean((tempH - np.dot(X, a)) ** 2))
        b = np.asarray(b, dtype=self.precision)

        devX = gpuarray.to_gpu(X)
        deva = gpuarray.to_gpu(a)
        devb = gpuarray.to_gpu(b)
        newH = self.func["sigm"](devX, deva, devb).get()
        newH = self.H_scaler.inverse_transform(newH)
        newB = np.dot(np.linalg.pinv(newH), delta)

        return a, b, newB

    def solve_bslfn(self, X, T):
        weights = np.zeros((self.inputs, 1))
        bias = np.zeros(1)
        beta = np.zeros((1, 1))
        count = 1
        while count <= self.L:
            # generate input weights and bias
            a0 = np.random.randn(self.inputs, 1)
            a0 *= 3.0 / self.inputs ** 0.5
            b0 = np.random.randn(1)

            delta = self._delta(X, a0, b0, T)

            from sklearn.preprocessing import MinMaxScaler
            self.H_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
            B_pinv = np.linalg.pinv(self.B)
            tempH = self.H_scaler.fit_transform(np.dot(delta, B_pinv))

            a, b, B = self.update_weights(tempH, X, delta)

            # save weights, bias , H and beta(output weights)
            weights = np.hstack((weights, a))
            bias = np.hstack((bias, b))
            beta = np.vstack((beta, B))
            count += 1
        self.neurons = [(self.L, "sigm", weights[:, 1:], bias[1:])]
        self.B = beta[1:, :]

    def _project(self, X, dev=False):
        """Projects X to H, an auxiliary function that implements a particular projection.
        For actual projection, use `ELM.project()` instead.
        Args:
            X (matrix): an input data matrix, size (N * `inputs`)
            dev (bool, optional): whether leave result in the GPU memory
        Returns:
            H (matrix): an SLFN hidden layer representation, size (N * `L`) where 'L' is number of neurons
        """
        assert self.neurons is not None, "ELM has no neurons"
        X = np.array(X, order="C", dtype=self.precision)
        devX = gpuarray.to_gpu(X)
        devH = gpuarray.empty((X.shape[0], self.L), dtype=self.precision)
        i = 0
        for nn, ftype, W, B in self.neurons:
            devW = gpuarray.to_gpu(W.astype(self.precision))
            devB = gpuarray.to_gpu(B.astype(self.precision))
            devH[:, i:i + nn] = self.func[ftype](devX, devW, devB)
            i += nn

        H = devH if dev else devH.get()
        return H

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
