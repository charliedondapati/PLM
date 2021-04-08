from hpelm import ELM
# from hpelm.nnets import SLFN
from six import integer_types
from BELM.net.bslfn import BSLFN


class BELM(ELM):
    """ """

    def __init__(self, inputs, outputs, classification="", w=None, batch=1000, accelerator=None,
                 precision='double', norm=None, tprint=5):
        super().__init__(inputs, outputs, classification, w, batch, accelerator, precision, norm, tprint)

        # invoking BSLFN nnet class
        if accelerator is "GPU":
            print("Using CUDA GPU acceleration with Scikit-CUDA")
            from BELM.net.bslfn_skcuda import BSLFNskCUDA
            self.nnet = BSLFNskCUDA(inputs, outputs, precision=self.precision, norm=norm)
        else:
            # print("Using slower basic Python solver")
            self.nnet = BSLFN(inputs, outputs, precision=self.precision, norm=norm)

        self.flist = ("sigm")

    def add_neurons(self, number, func, W=None, B=None):
        """
        """
        assert isinstance(number, integer_types), "Number of neurons must be integer"
        assert (func in self.flist), \
            "'%s' neurons not suppored: use a standard neuron function or a custom <numpy.ufunc>" % func
        # print("in belm")
        self.nnet.add_neurons(number, func, W, B)

    def train(self, X, T, *args, **kwargs):
        """ """

        X, T = self._checkdata(X, T)
        self._train_parse_args(args, kwargs)

        # only regression
        self.nnet.solve_bslfn(X, T)

    def predict(self, X):
        """Predict outputs Y for the given input data X.
        Args:
            X (matrix): input data of size (N * `inputs`)
        Returns:
            Y (matrix): output data or predicted classes, size (N * `outputs`).
        """
        X, _ = self._checkdata(X, None)
        Y = self.nnet._predict_belm(X)
        return Y
