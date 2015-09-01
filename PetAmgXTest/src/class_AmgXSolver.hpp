# include "headers.hpp"
# include "cudaCHECK.hpp"
# include <amgx_c.h>
# include <cuda_runtime.h>

class AmgXSolver
{
    public:

        AmgXSolver() = default;

        int initialize(MPI_Comm comm, int _Npart, int _myRank,
                const std::string &_mode, const std::string &cfg_file);

        int finalize();

        int setA(Mat &A);
        int solve(Vec &p, Vec &b);


    private:

        static int              count;      // only one instance allowed

        bool                    isInitialized = false,  // as its name
                                isUploaded_A = false,   // as its name
                                isUploaded_P = false,   // as its name
                                isUploaded_B = false;   // as its name

        int                     Ndevs,      // # of cuda devices
                                Npart,      // # of partitions
                                myRank,     // rank of current process
                                ring;       // a parameter used by AmgX

        int                    *devs = nullptr;     // list of devices used by
                                                    // current process


        MPI_Comm                AmgXComm;   // communicator
        AMGX_Mode               mode;       // AmgX mode
        AMGX_config_handle      cfg;        // AmgX config object
        AMGX_resources_handle   rsrc;       // AmgX resource object
        AMGX_matrix_handle      AmgXA;      // AmgX coeff mat
        AMGX_vector_handle      AmgXP,      // AmgX unknowns vec
                                AmgXRHS;    // AmgX RHS vec
        AMGX_solver_handle      solver;     // AmgX solver object


        int setMode(const std::string &_mode);
        int getPartVec(const Mat &A, int *& partVec);
        static void print_callback(const char *msg, int length);
        static void print_none(const char *msg, int length);

};
