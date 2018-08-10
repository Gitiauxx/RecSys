import pandas as pd
from numba import jit
import numpy as np


class matrix_factorization(object):

    def __init__(self, R, n_users, n_items, k=10, eta=0.05, beta=0.01):

        self.R = R
        self.eta = eta
        self.beta = beta
        self.n = len(self.R)
        self.U =  np.random.normal(scale=1 / k, size=(n_users, k))
        self.V = np.random.normal(scale=1 / k, size=(n_items, k))

    def gradient_descent(self, R):

        for i in range(R.shape[0]):
            user = int(R[i, 0])
            item = int(R[i, 1])
            error = R[i, 2] - np.dot(self.U[user, :], self.V[item, :].transpose())
            update_u = self.V[item, :] * error - \
                              self.beta * self.U[user, :]
            update_v = self.U[user, :] * error - \
                              self.beta * self.V[item, :]
            self.U[user, :] += self.eta * update_u
            self.V[item, :] += self.eta * update_v

            self.squared_error += error**2

    def fit(self, nepoch=10):

        for epoch in range(nepoch):
            self.squared_error = 0
            np.random.shuffle(self.R)
            self.gradient_descent(self.R)
            
    def predict(self, testX):
        
        size = testX.shape[0]
        testY = np.zeros(size)
        for i in range(size):
            user = int(testX[i, 0])
            item = int(testX[i, 1])
            testY[i]= self.get_rating_item_user(user, item)

			
        return testY

    def get_rating_item_user(self, user, item):
        return np.dot(self.U[user, :], self.V[item, :].transpose())

    def get_rating(self):
        return np.dot(self.U, self.V.T)

class matrix_factorization_bias(matrix_factorization):

    def __init__(self, R, n_users, n_items, k, eta, beta):
        super().__init__(R, n_users, n_items, k, eta, beta)
        self.A = np.random.normal( scale=1, size=n_users) + 2
        self.B = np.random.normal(scale=1, size=n_items) + 2
		
		# momentum
        self.UM = np.zeros((n_users, k))
        self.IM = np.zeros((n_items, k))
        self.AM = np.zeros(n_users)
        self.BM = np.zeros(n_items)
		
        self.momentum = 0.8


    def gradient_descent(self, R):

        for i in range(R.shape[0]):
            user = int(R[i, 0])
            item = int(R[i, 1])
            error = R[i, 2] - np.dot(self.U[user, :], self.V[item, :].transpose()) - \
                    self.A[user] - self.B[item]

            update_u = self.V[item, :] * error - \
                              self.beta * self.U[user, :]
            update_v = self.U[user, :] * error - \
                              self.beta * self.V[item, :]

            update_a = error - self.beta * self.A[user]
            update_b = error - self.beta * self.B[item]

            self.U[user, :] += self.eta * update_u
            self.V[item, :] += self.eta * update_v
            self.A[user] += self.eta * update_a
            self.B[item] += self.eta * update_b

            self.squared_error += error**2
		

    def get_rating_item_user(self, user, item):
        return np.dot(self.U[user, :], self.V[item, :].transpose()) + self.A[user] + self.B[item]


class matrix_factorization_numba(matrix_factorization_bias):
    def gradient_descent(self, R):
        self.squared_error, self.U, self.V, self.A, self.B, self.UM, self.IM, self.AM, self.BM =_gradient_descent(R, self.U, self.V, self.A,
                                                                              self.B, self.eta, self.beta, self.UM, self.IM, self.AM, self.BM,
																			  )
        


@jit(nopython=True)
def _gradient_descent(R, U, V, A, B, eta, beta, UM, IM, AM, BM):

    squared_error = 0
    momentum = 0.8
	
    for i in range(R.shape[0]):
        user = int(R[i, 0])
        item = int(R[i, 1])
        error = R[i, 2] - np.dot(U[user, :], V[item, :].transpose()) - \
                A[user] - B[item]

        update_u = V[item, :] * error - beta * U[user, :]
        update_v = U[user, :] * error - beta * V[item, :]
        update_a = error - beta * A[user]
        update_b = error - beta * B[item]
		
		# update momentum
        UM[user, :] = momentum * UM[user, :] - eta * update_u
        IM[item, :] = momentum * IM[item, :] - eta * update_v
        AM[user] = momentum * AM[user] - eta * update_a
        BM[item] = momentum * BM[item] - eta * update_b
        
        U[user, :] -= UM[user, :]
        V[item, :] -= IM[item, :]
        A[user] -= AM[user]
        B[item] -= BM[item]
		
        squared_error += error**2

    return squared_error, U, V, A, B, UM, IM, AM, BM

@jit(nopython=True)
def _gradient_descent_plus(R, U, V, A, B, Y, eta, beta):
    squared_error = 0
    for i in range(R.shape[0]):
        user = int(R[i, 0])
        item = int(R[i, 2])
        error = R[i, 1] - np.dot(U[user, :] + Y[user, :], V[item, :].transpose()) - A[user] - B[item]

        update_u = V[item, :] * error - beta * U[user, :]
        update_v = U[user, :] * error - beta * V[item, :]
        update_a = error - beta * A[user]
        update_b = error - beta * B[item]

        U[user, :] += eta * update_u
        V[item, :] += eta * update_v
        A[user] += eta * update_a
        B[item] += eta * update_b
        squared_error += error ** 2

    return squared_error, U, V, A, B



if __name__ == '__main__':
    R = np.load('C:\\Users\\MX\\Documents\\Xavier\\CSPrel\\Recommneder\\netflix data\\short_training.npy')

    n_users = np.unique(R[:, 0]).shape[0]
    n_items = np.unique(R[:, 2]).shape[0]

    print("The number of items is {} and the number of users is {} \n".format(n_items, n_users))

    mf = matrix_factorization_numba_plus(R, n_users, n_items, 10, 0.05, 0.2)
    mf.iteration(10, method='MF_numba_bias')


