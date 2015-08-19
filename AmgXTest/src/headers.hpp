# pragma once

# include <iostream>
# include <fstream>
# include <vector>
# include <string>
# include <map>
# include <cmath>

# include <amgx_c.h>

# include <boost/program_options.hpp>
# include <boost/timer/timer.hpp>
# include <boost/mpi.hpp>

# include <cuda_runtime.h>

# include "SpMt.hpp"
# include "cudaCHECK.h"


# ifndef DMEMTYPE
# define DMEMTYPE
enum memType { HOST, DEVICE };
# endif


typedef std::map<std::string, std::string>      paramsMap;

typedef thrust::host_vector<double>             HostVec;
typedef thrust::device_vector<double>           DeviceVec;
typedef thrust::host_vector<int>                HostVecInt;
typedef thrust::device_vector<int>              DeviceVecInt;

typedef SpMt<HOST>                              HostSpMt;
typedef SpMt<DEVICE>                            DeviceSpMt;


void print_callback(const char *, int);

void parseCMD(int, char **, paramsMap &);

void setMode(std::string, AMGX_Mode &);

void generateA(const int &Nx, const int &Ny, 
               const int &Nlocal, const int &bgIdx,
               const double &dx, const double &dy, HostSpMt &A);

void generateXY(const int &Nx, const int &Ny,
                const double &Lx, const double &Ly,
                double &dx, double &dy, HostVec &x, HostVec &y);

void generateRHS(const int &Nx, const int &Ny, 
                 const int &Nlocal, const int &bgIdx,
                 HostVec &b, const HostVec &x, const HostVec &y);

void generateZerosVec(const int &N, HostVec &p);

void exactSolution(const int &Nx, const int &Ny,
                   const int &Nlocal, const int &bgIdx,
                   HostVec &p_exact, const HostVec &x, const HostVec &y);
    
double errL2Norm(const int &N, const HostVec &p, const HostVec &p_exact);

int showSysInfo(const AMGX_matrix_handle &A, 
                 const AMGX_vector_handle &b, const AMGX_vector_handle &x);

void checkError(const int Nx, const int Ny, 
                const HostVec &p, const HostVec &x, const HostVec &y);

int fetchMtxSize(std::string mtxFile);

void detPartSize(const int &Ntol, const int &mpiSize, const int &myRank,
                 int &Nlocal, int &bgIdx, int &edIdx, HostVecInt &partVec);

template<typename T>
std::ostream & operator<<(std::ostream &os, std::vector<T> x);
