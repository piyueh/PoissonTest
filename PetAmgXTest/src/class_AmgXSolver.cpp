# include "class_AmgXSolver.hpp"


bool AmgXSolver::count = false;


AmgXSolver::AmgXSolver()
{
    if (count)
    {
        std::cerr << "Only one AmgXSolver instance is allowed "
                  << " in one process!!" << std::endl;
        exit(0);
    }
    else
        count = true;
}


AmgXSolver::~AmgXSolver()
{
    count = false;
}


int AmgXSolver::initialize(MPI_Comm comm, int _Npart, int _myRank,
        const std::string &_mode, const std::string &cfg_file)
{
    AmgXComm = comm;
    Npart = _Npart;
    myRank = _myRank;

    CHECK(cudaGetDeviceCount(&Ndevs));
    devs = new int(myRank % Ndevs);

    // initialize AmgX
    MPI_Barrier(AmgXComm);
    AMGX_SAFE_CALL(AMGX_initialize());

    // intialize AmgX plugings
    MPI_Barrier(AmgXComm);
    AMGX_SAFE_CALL(AMGX_initialize_plugins());

    // use user define output mechanism
    // (to be finished ...)

    // let AmgX to handle errors returned
    MPI_Barrier(AmgXComm);
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    // set mode

    
    return 0;
}
