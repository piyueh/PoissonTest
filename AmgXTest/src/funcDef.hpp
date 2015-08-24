# include "headers.hpp"


void print_callback(const char *, int);

void print_none(const char *, int);

void parseCMD(int, char **, paramsMap &);

void setMode(std::string, AMGX_Mode &);

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

# include "generateA.tcc"
