# include "headers.hpp"
# include "checkCUDA.hpp"
# include <amgx_c.h>
# include <cuda_runtime.h>

class AmgXSolver
{
    public:

        AmgXSolver();

        ~AmgXSolver();

        int initialize(const std::string &_mode, 
                const std::string &cfg_file, MPI_Comm comm);



    private:

        static bool             count;      // only one instance allowed

        bool                    isInitialized = false,  // as its name
                                isUploaded_A = false,   // as its name
                                isUploaded_P = false,   // as its name
                                isUploaded_B = false;   // as its name

        AMGX_Mode               mode;       // AmgX mode
        AMGX_config_handle      cfg;        // AmgX config object
        AMGX_resources_handle   rsrc;       // AmgX resource object
        AMGX_matrix_handle      amgxA;      // AmgX coeff mat
        AMGX_vector_handle      amgxP,      // AmgX unknowns vec
                                amgxRHS;    // AmgX RHS vec
        AMGX_solver_handle      solver;     // AmgX solver object

        struct
        {
            int         Nrows,      // # of rows in this process
                        Nnz;        // # on non-zero entries
            int        *row;        // row indicies
            long       *col;        // col indicies
            double     *data;       // entries
        }                       RawMat;     // containing raw data of matrix
        
        int                     Ndevs;      // # of cuda devices
        int                    *devs = nullptr;     // list of devices used by
                                                    // current process

        int                     Npart,      // # of partitions
                                myRank;
        int                    *partVec = nullptr;  // list of partition
        MPI_Comm                AmgXComm;   // communicator
};
