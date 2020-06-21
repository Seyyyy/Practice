import numpy as np
from scipy import linalg

class LinearRegression:
    def __init__(self):
        self.w_ = None


    def fit(self, X, t):
        '''
        X: 入力訓練データ
        t: 出力訓練データ
        Xtil: 入力訓練データに切片成分のベクトルを追加したもの
        '''
        Xtil = np.c_[np.ones(X.shape[0]), X]
        A = np.dot(Xtil.T, Xtil)
        b = np.dot(Xtil.T, t)
        self.w_ = linalg.solve(A, b)


    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)