# pragma once

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


template<memType T>
class SpMt{};


template<>
class SpMt<HOST>: public SpMt_Base
{
    public:
        thrust::host_vector<int>        rowIdx,
                                        colIdx;
        thrust::host_vector<double>     data;

        SpMt() = default;
        SpMt & operator=(const SpMt<HOST> &rhs);
        SpMt & operator=(const SpMt<DEVICE> &rhs);
};


template<>
class SpMt<DEVICE>: public SpMt_Base
{
    public:

        thrust::device_vector<int>      rowIdx,
                                        colIdx;
        thrust::device_vector<double>   data;

        SpMt() = default;
        SpMt & operator=(const SpMt<HOST> &rhs);
        SpMt & operator=(const SpMt<DEVICE> &rhs);
};
