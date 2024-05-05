#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from dotmap import DotMap
from loguru import logger
from pytictoc import TicToc


class DataLoader(object):
    def __init__(self,
                 random_seed=0,
                 reduction_dimension=128,
                 reduction_iterations=30):
        """
        Constructor of DataLoader

        :param random_seed: random seed
        :param reduction_dimension: input feature dimension (SVD)
        :param reduction_iterations: number of iterations required by SVD
        """

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.reduction_dimension = reduction_dimension
        self.reduction_iterations = reduction_iterations
        self.timer = TicToc()

    def load(self, data_path, test = False):
        """
        Load data and split the data into training and test.

        :param data_path: path for dataset
        :param heldout_ratio: heldout ratio between training and test
        :return: loaded data
        """
        logger.info('Start loading the signed network...')
        val_X = np.loadtxt(data_path+"_val.txt", dtype='int', delimiter='\t')
        val_y = val_X[:, 2]
        testing_X = np.loadtxt(data_path+"_test.txt", dtype='int', delimiter='\t')
        testing_y = testing_X[:, 2]
        
        if test == False:
            test_X = val_X
            test_y = val_y
        if test == True:
            test_X = testing_X
            test_y = testing_y
        train_X = np.loadtxt(data_path+"_training.txt", dtype='int', delimiter='\t')
        train_y = train_X[:, 2]

        num_nodes = max(np.amax(train_X[:, 0:2]) + 1, np.amax(val_X[:, 0:2]) + 1, np.amax(testing_X[:, 0:2]) + 1)
        num_edges = train_X.shape[0] + val_X.shape[0] + testing_X.shape[0]
        logger.info('Start creating input features with random_seed: {}...'.format(self.random_seed))
        self.timer.tic()
        H = self.generate_input_features(train_X, num_nodes)

        gen_time = self.timer.tocvalue()
        logger.info('Generation input features completed in {:.4} sec'.format(gen_time))

        data = DotMap()
        data.train.X = train_X
        data.train.y = train_y
        data.test.X = test_X
        data.test.y = test_y
        data.H = H                   # input feature matrix
        data.num_nodes = num_nodes

        neg_idx = train_X[:, 2] < 0
        neg_ratio = train_X[neg_idx, :].shape[0] / float(train_X.shape[0])
        data.neg_ratio = neg_ratio
        data.class_weights = np.asarray([1.0, 1.0])

        return data

    def generate_input_features(self, train_edges, num_nodes):
        """
        Create spectral features based on SVD
        :return: SVD input features
        """
        src = train_edges[:, 0]
        dst = train_edges[:, 1]
        sign = train_edges[:, 2]
        shaping = (num_nodes, num_nodes)
        signed_A = sparse.csr_matrix((sign, (src, dst)),
                                     shape=shaping,
                                     dtype=np.float32)

        svd = TruncatedSVD(n_components=self.reduction_dimension,
                           n_iter=self.reduction_iterations,
                           random_state=self.random_seed)

        X = svd.fit_transform(signed_A)  # equivalent to U * Sigma

        return X
