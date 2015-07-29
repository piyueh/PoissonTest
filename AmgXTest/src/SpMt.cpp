# include "cudaCHECK.h"
# include "SpMt.hpp"
# include <cuda.h>


SpMt<HOST> & SpMt<HOST>::operator=(const SpMt<HOST> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


SpMt<HOST> & SpMt<HOST>::operator=(const SpMt<DEVICE> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


SpMt<DEVICE> & SpMt<DEVICE>::operator=(const SpMt<HOST> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}


SpMt<DEVICE> & SpMt<DEVICE>::operator=(const SpMt<DEVICE> &rhs)
{
    Nrows = rhs.Nrows;
    Nnz = rhs.Nnz;

    rowIdx     = rhs.rowIdx;
    colIdx     = rhs.colIdx;
    data       = rhs.data;

    return *this;
}
