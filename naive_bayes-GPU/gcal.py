import time
import numpy as np
import pandas as pd
import math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def o_matmul_gpu(data):

    mod = SourceModule("""
        const int TILE_WIDTH = 32;
        const int ONE = 1;
        __global__ void matrix_mul(float *A, float *B, float *C,
                                   int numARows, int numAColumns) {
        
            __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
            __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];
            float v = 0.0;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = by*TILE_WIDTH + ty;
            int col = bx*TILE_WIDTH + tx;
            int size = numAColumns%TILE_WIDTH==0 ? numAColumns/TILE_WIDTH : numAColumns/TILE_WIDTH+1;
            
            for (int i = 0; i < size; i++)
            {
                if (i*TILE_WIDTH + tx < numAColumns && row < numARows) {
                    sharedM[ty][tx] = A[row*numAColumns + i*TILE_WIDTH + tx];
                }
                    
                else {
                    sharedM[ty][tx] = 0.0;
                }
                    

                if (i*TILE_WIDTH + ty < numAColumns && col < ONE)
                    sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*ONE + col];
                else
                    sharedN[ty][tx] = 0.0;
                __syncthreads();

                for(int j = 0; j < TILE_WIDTH; j++) {
                    v += sharedM[ty][j] * sharedN[j][tx];
                }
                    
                __syncthreads();
            }

            if (row < numARows && col < ONE) {
                C[row*ONE + col] = v;
            }
        }
    """)
    # print(data)
    data.astype('float32')
    A = np.copy(data.values.T)
    # A = np.copy(AT.T)
    rows, cols = np.int32(A.shape)
    B = np.ones(cols, dtype=np.float32)
    C = np.zeros(rows, dtype=np.float32)

    matrix_mul = mod.get_function('matrix_mul')
    A = np.float32(A)
    one = np.int32(1)
    thread_size = 32
    grid_x = 1
    grid_y = int(math.ceil(rows/thread_size))
    start = time.time()
    matrix_mul(
        cuda.In(A), cuda.In(B), cuda.Out(C), rows, cols, cols, one, rows, one,
        block=(thread_size, thread_size, 1), grid=(grid_x, grid_y)
    )
    # print(f'gpu run time: {time.time() - start}')
    # print('-'*20)
    # print(data.mean())
    # print(C / cols)
    # print(np.matmul(A, B))
    return C / cols


def d_matmul_gpu(data, data1):

    mod = SourceModule("""

        const int TILE_WIDTH = 32;
        const int ONE = 1;
        __global__ void matrix_mul(float *A, float *B, float *C, float *A1, float *C1,
                                   int numARows, int numAColumns) {
        
        __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
        __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];
        __shared__ float sharedM1[TILE_WIDTH][TILE_WIDTH];
        float v = 0.0;
        float v1 = 0.0;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by*TILE_WIDTH + ty;
        int col = bx*TILE_WIDTH + tx;
        int size = numAColumns%TILE_WIDTH==0 ? numAColumns/TILE_WIDTH : numAColumns/TILE_WIDTH+1;
        
        for (int i = 0; i < size; i++)
        {
            if (i*TILE_WIDTH + tx < numAColumns && row < numARows) {
                sharedM[ty][tx] = A[row*numAColumns + i*TILE_WIDTH + tx];
                sharedM1[ty][tx] = A1[row*numAColumns + i*TILE_WIDTH + tx];
            }
                
            else {
                sharedM[ty][tx] = 0.0;
                sharedM1[ty][tx] = 0.0;
            }
                

            if (i*TILE_WIDTH + ty < numAColumns && col < ONE)
                sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*ONE + col];
            else
                sharedN[ty][tx] = 0.0;
            __syncthreads();

            for(int j = 0; j < TILE_WIDTH; j++) {
                v += sharedM[ty][j] * sharedN[j][tx];
                v1 += sharedM1[ty][j] * sharedN[j][tx];
            }
                
            __syncthreads();
        }

        if (row < numARows && col < ONE) {
            C[row*ONE + col] = v;
            C1[row*ONE + col] = v1;
        }
            
        
    }

    """)
    # print(data)
    data.astype('float32')
    data1.astype('float32')
    add_rows = 0
    add_item = np.zeros(data.shape[1], dtype=np.float32)
    if data.shape[0] >= data1.shape[0]:
        A = np.copy(data.values.T)
        A1 = np.copy(data1.values.T)
    else:
        A1 = np.copy(data1.values.T)
        A = np.copy(data.values.T)

    rowsA, colsA = A.shape
    add_rows = rowsA - A1.shape[0]
    for i in range(add_rows):
        A1 = np.insert(A1, 0, values=add_item, axis=0)

    # A = np.copy(AT.T)
    rows, cols = np.int32(A.shape)
    B = np.ones(cols, dtype=np.float32)
    C = np.zeros(rows, dtype=np.float32)
    C1 = np.zeros(rows, dtype=np.float32)

    matrix_mul = mod.get_function('matrix_mul')
    A = np.float32(A)
    A1 = np.float32(A1)
    thread_size = 32
    grid_x = 1
    grid_y = int(math.ceil(rows/thread_size))
    start = time.time()
    matrix_mul(
        cuda.In(A), cuda.In(B), cuda.Out(C), cuda.In(A1), cuda.Out(C1), rows, cols,
        block=(thread_size, thread_size, 1), grid=(grid_x, grid_y)
    )
    for i in range(add_rows):
        C1 = np.delete(C1, 0, 0)
    # print(f'gpu run time: {time.time() - start}')
    
    return C / colsA, C1 / colsA


def bayes_gpu(data, means, stds):

    mod = SourceModule("""
        #define _USE_MATH_DEFINES
        #include <math.h>
        const int TILE_WIDTH = 32;
        __global__ void probability(float *A, float *B, float *B1, float *C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns) {
        
            __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
            __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];
            __shared__ float sharedN1[TILE_WIDTH][TILE_WIDTH];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = by*TILE_WIDTH + ty;
            int col = bx*TILE_WIDTH + tx;
            float v = 1.0;

            for (int i = 0; i < (int)(ceil((float)numAColumns/TILE_WIDTH)); i++)
            {
                if (i*TILE_WIDTH + tx < numAColumns && row < numARows)
                    sharedM[ty][tx] = A[row*numAColumns + i*TILE_WIDTH + tx];
                else
                    sharedM[ty][tx] = 0.0;

                if (i*TILE_WIDTH + ty < numBRows && col < numBColumns) {
                    
                    sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*numBColumns + col];
                    sharedN1[ty][tx] = B1[(i*TILE_WIDTH + ty)*numBColumns + col];
                }
                else {
                    sharedN[ty][tx] = 0.0;
                    sharedN1[ty][tx] = 1.0;
                }
                __syncthreads();
                //float a_t = 0.0;
                //float b_t = 0.0;
                for(int j = 0; j < TILE_WIDTH; j++) {
                    //v *= sharedM[ty][j] * sharedN[j][tx] * sharedN1[j][tx];
                    //v *= sharedM[ty][j];
                    //float a = pow((sharedM[ty][j]-sharedN[j][tx]), 2);
                    //a_t = exp(-1*a/(sharedN1[j][tx]*2));
                    //b_t = sqrt(2*M_PI*sharedN1[j][tx]);
                    //printf("%f ", a);
                    //v *= a_t / b_t;
                    v *= exp(-1*pow((sharedM[ty][j]-sharedN[j][tx]), 2)/(sharedN1[j][tx]*2))/sqrt(2*M_PI*sharedN1[j][tx]);
                }
                    
                __syncthreads();
            }

            if (row < numCRows && col < numCColumns)
                C[row*numCColumns + col] = v;
            
        }
    """)
    # print(data)
    data.astype('float32')
    means.astype('float32')
    stds.astype('float32')
    # print(data, means, stds)
    A = np.copy(data.values, order='C')
    B = np.copy(means.values.T)
    B1 = np.copy(stds.values.T)
    # A = np.copy(data, order='C')
    # B = np.copy(means.T, order='C')
    # B1 = np.copy(stds.T, order='C')
    rowsA, colsA = np.int32(A.shape)
    rowsB, colsB = np.int32(B.shape)
    C = np.zeros((rowsA, colsB), dtype=np.float32)

    matrix_mul = mod.get_function('probability')
    A = np.float32(A)
    B = np.float32(B)
    B1 = np.float32(B1)
    thread_size = 32
    grid_x = int(math.ceil(colsB/thread_size))
    grid_y = int(math.ceil(rowsA/thread_size))
    start = time.time()
    matrix_mul(
        cuda.In(A), cuda.In(B), cuda.In(B1), cuda.Out(C), rowsA, colsA, rowsB, colsB, rowsA, colsB,
        block=(thread_size, thread_size, 1), grid=(grid_x, grid_y)
    )
    # print(f'gpu run time: {time.time() - start}')
    # print(C)
    # print(np.matmul(A, B1))
    return C


def bayes_gpu1(data, means, stds):

    mod = SourceModule("""
        #define _USE_MATH_DEFINES
        #include <math.h>
        __global__ void probability(float *A, float *B, float *B1, float *C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns) 
        {
            float sum = 1.0;
            int row = blockIdx.y*blockDim.y + threadIdx.y;
            int col = blockIdx.x*blockDim.x + threadIdx.x;
            if(row < numCRows && col < numCColumns){
                for (int i = 0; i < numAColumns; ++i) {
                    //sum *= A[row*numAColumns + i] * B[i*numBColumns + col];
                    sum *= exp(-1*pow(A[row*numAColumns + i]-B[i*numBColumns + col], 2)/(B1[i*numBColumns + col]*2))/sqrt(2*M_PI*B1[i*numBColumns + col]);
                }
                C[row*numBColumns + col] = sum;
            }
        }
    """)
    # print(data)
    data.astype('float32')
    means.astype('float32')
    stds.astype('float32')
    # print(data, means, stds)
    A = np.copy(data.values, order='C')
    B = np.copy(means.values.T)
    B1 = np.copy(stds.values.T)
    # A = np.copy(data, order='C')
    # B = np.copy(means.T, order='C')
    # B1 = np.copy(stds.T, order='C')
    rowsA, colsA = np.int32(A.shape)
    rowsB, colsB = np.int32(B.shape)
    C = np.zeros((rowsA, colsB), dtype=np.float32)

    matrix_mul = mod.get_function('probability')
    A = np.float32(A)
    B = np.float32(B)
    B1 = np.float32(B1)
    thread_size = 32
    grid_x = int(math.ceil(colsB/thread_size))
    grid_y = int(math.ceil(rowsA/thread_size))
    # start = time.time()
    matrix_mul(
        cuda.In(A), cuda.In(B), cuda.In(B1), cuda.Out(C), rowsA, colsA, rowsB, colsB, rowsA, colsB,
        block=(thread_size, thread_size, 1), grid=(grid_x, grid_y)
    )
    # print(f'gpu run time: {time.time() - start}')
    # print(C)
    # print(np.matmul(A, B1))
    return C

