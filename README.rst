Progressive Learning Machine
----------------------------


Description: 
 In neural networks(NNs), the universal approximation capability of NNs is widely used in many applications. However, this property is generally proven for continuous systems. Most industrial systems are hybrid systems , which is a significant limitation for real applications. Progressive Learning Machine is a proven Multi-NN for general hybrid nonlinear or linear system approximation. PLM classifies hybrid systems into several continuous systems and performs approximation to calculate output error.

Development:
 The Structure of PLM has 2 main stages:
- Training Data: cluster growing and optimal offspring cluster selection
- Testing Data:  cluster estimation for testing data points
  
 Bi-directional ELM(B-ELM) is used in PLM to train multiple neural networks on clusters of several continous systems and optimal NN for each cluster is selected. These selected B-ELM NNs are used on clusters of test data.

Dataset:
 Based on the dataset functions presented in [1], datasets are generated. Datasets should be processed as below:
- Data: Input features are arranged in column before Output
- Output: Input features are followed by output for each row in the column
- Label: Output feature is followed by the cluster label


Main module & function:
 - PLM.pelm is the main module used to call train and test functions
 - pelm() is the wrapper function that calls train and test functions internally. The number of epochs can be specified as a parameter, the default is epochs=20.

Installation:
 >>> pip install PLM


Example usage:
  >>> from PLM.pelm import pelm
  >>> n = 10;
  >>> parameter1 = 10;
  >>> parameter2 = 10;
  >>> model_number = 3;
  >>> pelm(data, target, model_number, n=n, p=parameter1, s=parameter2)

Authors & Acknowledgements:
- See the included AUTHORS file for more information.
- Special thanks to the author or the paper [1] - Yimin Yang, Comp. Sc, Lakehead University.
  
License:
 This software is licensed under the BSD License. See the included LICENSE file for more information.


References:

1. Yimin Yang, Yaonan Wang, Q.M.Jonathan Wu et al. ”Progressive Learning Machine: A New Approach for General Hybrid System Approximation”. IEEE Transactions on Neural Networks and Learning Systems. Vol. 26, pp. 1855 - 1874, 2015.
2. Yimin Yang, Wang Yaonan, et al. "Bidirectional extreme learning machine for regression problem and its learning effectiveness". IEEE Transactions on Neural Networks and Learning Systems. 23(9), pp. 1498 - 1505. 2012.
3. https://www.ntu.edu.sg/home/egbhuang/
4. https://docs.nvidia.com/cuda/
5. https://towardsdatascience.com/introduction-to-extreme-learning-machines-c020020ff82b
