# pragma once

# include "cudaCHECK.h"
# include <cuda_runtime.h>
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>


# ifndef DMEMTYPE
# define DMEMTYPE
enum memType { HOST, DEVICE };
# endif


class SpMt_Base
{
    public:
        int         Nrows=0, 
                    Nnz=0;
};


template<memType M, typename T>
class SpMt{};


template<typename T>
class SpMt<HOST, T>: public SpMt_Base
{
    public:
        thrust::host_vector<int>        rowIdx;
        thrust::host_vector<T>          colIdx;
        thrust::host_vector<double>     data;

        SpMt() = default;
        SpMt & operator=(const SpMt<HOST, T> &rhs);
        SpMt & operator=(const SpMt<DEVICE, T> &rhs);
};


template<typename T>
class SpMt<DEVICE, T>: public SpMt_Base
{
    public:

        thrust::device_vector<int>      rowIdx;
        thrust::device_vector<T>        colIdx;
        thrust::device_vector<double>   data;

        SpMt() = default;
        SpMt & operator=(const SpMt<HOST, T> &rhs);
        SpMt & operator=(const SpMt<DEVICE, T> &rhs);
};

