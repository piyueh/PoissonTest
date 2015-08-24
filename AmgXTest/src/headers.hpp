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

typedef SpMt<HOST, int>                         HostSpMtInt;
typedef SpMt<DEVICE, int>                       DeviceSpMtInt;
typedef SpMt<HOST, long>                        HostSpMtLong;
typedef SpMt<DEVICE, long>                      DeviceSpMtLong;


