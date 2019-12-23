import math
import numpy as np
import operator

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


mod = SourceModule("""
    const int TILE_WIDTH = 32;
    __global__ void distance(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
    
    float sum = 0.0;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < numCRows && col < numCColumns){
        for (int i = 0; i < numAColumns; ++i)
        {
            sum += (A[row*numAColumns + i]-B[i*numBColumns + col])*(A[row*numAColumns + i]-B[i*numBColumns + col]);
        }
        C[row*numBColumns + col] = sum;
    }
    
}

""")


class KNN(object):

    def __init__(self, k=3):
        self.k = k
        self.thread_size = 32


    def fit(self, x, y):
        self.x = np.float32(np.copy(x.T, order='C'))
        self.y = np.int32(y)

    def _vote(self, ys):
        ys_unique = np.unique(ys)
        # print('ys_unique, ys: %d %d'%(len(ys_unique), len(ys)))
        vote_dict = {}
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_vote_dict[0][0])
        return sorted_vote_dict[0][0]


    def predict(self, x):
        x = np.float32(x, order='C')
        y_pred = []
        distance = mod.get_function('distance')
        train_row, train_col = np.int32(self.x.shape)
        test_row, test_col = np.int32(x.shape)
        grid_x = int(math.ceil(train_col/self.thread_size))
        grid_y = int(math.ceil(test_row/self.thread_size))
        # print(grid_x, grid_y)
        dis_arr = np.zeros((test_row, train_col), dtype=np.float32)
        distance(
            cuda.In(x), cuda.In(self.x), cuda.Out(dis_arr), test_row, test_col, train_row, train_col, test_row, train_col,
            block=(self.thread_size, self.thread_size, 1), grid=(grid_x, grid_y)
        )
        # print(dis_arr)
        
        for i in range(len(dis_arr)):
            sorted_index = np.argsort(dis_arr[i])
            top_k_index = sorted_index[:self.k]
            y_pred.append(self._vote(ys=self.y[top_k_index]))

        return np.array(y_pred)

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score = 0.0
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                score += 1
        score /= len(y_true)
        return score