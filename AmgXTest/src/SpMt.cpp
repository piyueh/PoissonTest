# include "cudaCHECK.h"
# include "SpMt.hpp"
# include <cuda.h>


template<>
SpMt<HOST, int> & SpMt<HOST, int>::operator=(const SpMt<HOST, int> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<HOST, int> & SpMt<HOST, int>::operator=(const SpMt<DEVICE, int> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<HOST, long> & SpMt<HOST, long>::operator=(const SpMt<HOST, long> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<HOST, long> & SpMt<HOST, long>::operator=(const SpMt<DEVICE, long> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<DEVICE, int> & SpMt<DEVICE, int>::operator=(const SpMt<HOST, int> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<DEVICE, int> & SpMt<DEVICE, int>::operator=(const SpMt<DEVICE, int> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<DEVICE, long> & SpMt<DEVICE, long>::operator=(const SpMt<HOST, long> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


template<>
SpMt<DEVICE, long> & SpMt<DEVICE, long>::operator=(const SpMt<DEVICE, long> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}
