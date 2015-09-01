# include "headers.hpp"
# include "funcDef.hpp"


int main(int argc, char **argv)
{
    using namespace boost;


    // initialize mpi
    mpi::environment            env{argc, argv};
    mpi::communicator           world;



    // create a black hole that will be use by the output of other ranks
    std::basic_ostream<char>    blackHole(nullptr);
    std::ostream                &out = (world.rank() == 0) ? std::cout : blackHole;



    // a map containning user-input parameters through CMD
    paramsMap                   CMDparams;

    // parse CMD parameters and save into CMDparams
    parseCMD(argc, argv, CMDparams);

    // print out the user-input parameters
    std::cout << "Input parameters: " << std::endl;
    if (CMDparams.count("input"))
        out << "\t" << "input: " << CMDparams["input"] << std::endl;
    else
        for(auto it: CMDparams) out << "\t" << it.first << ": " << it.second << std::endl;



    // definition of original data
    // scalar parameters
    int                         Ntol,
                                Nx, 
                                Ny;
    double                      Lx,
                                Ly;
    double                      dx,
                                dy;


    // containers
    HostVec                     x,
                                y;
    HostSpMtLong                hA;
    HostVec                     hp,
                                hb;
    DeviceSpMtLong              dA;
    DeviceVec                   dp,
                                db;

    // raw pointers
    double                      *b,
                                *p,
                                *Adata;
    int                         *Arow;
    long                        *Acol;


    // variables for distributed system
    int                         myRank = world.rank(),
                                Npart = world.size(),
                                &NpartVec = Ntol,
                                Ndevs;
    HostVecInt                  partSize(Npart),
                                partVec,
                                devs;
    int                         &Nlocal = partSize[myRank],
                                bgIdx,
                                edIdx;


    // definition of all data related to AmgX
    AMGX_Mode                   mode;
    AMGX_config_handle          cfg;
    AMGX_resources_handle       rsrc;
    AMGX_matrix_handle          AmgX_A;
    AMGX_vector_handle          AmgX_p,
                                AmgX_b;
    AMGX_solver_handle          solver;
    AMGX_SOLVE_STATUS           status;


    // variabled for timer
    timer::cpu_timer            timer;
    std::string                 uploadTime,
                                setupTime,
                                solveTime,
                                downloadTime;




    // obtain the number of devices and set the device used by this rank
    CHECK(cudaGetDeviceCount(&Ndevs));
    devs.push_back(myRank % Ndevs);


    if (CMDparams.count("input"))
    {

        // get the total number of rows
        NpartVec = fetchMtxSize(CMDparams["input"]);
        world.barrier();


        // generate the vector partSize
        {
            int     Nbasic = Ntol / Npart;
            int     Nremain = Ntol % Npart;
            for(int i=0; i<Npart; ++i) partSize[i] = Nbasic;
            for(int i=0; i<Nremain; ++i) partSize[i] += 1;
        }
        world.barrier();


        // generate the vector partVec
        {
            partVec.resize(NpartVec);

            int     bg = 0, ed = 0;
            for(int i=0; i<Npart; ++i)
            {
                ed += partSize[i];
                for(int j=bg; j<ed; ++j) partVec[j] = i;
                bg = ed;
            }
        }
        world.barrier();
    }
    else
    {
        // initialize problem
        Nx = std::stoi(CMDparams["Nx"]);
        Ny = std::stoi(CMDparams["Ny"]);
        Ntol = Nx * Ny;

        Lx = 1.0;
        Ly = 1.0;

        detPartSize(Ntol, Npart, myRank, Nlocal, bgIdx, edIdx, partVec);

        generateXY(Nx, Ny, Lx, Ly, dx, dy, x, y);
        generateRHS(Nx, Ny, Nlocal, bgIdx, hb, x, y);
        generateZerosVec(Nlocal, hp);
        generateA(Nx, Ny, Nlocal, bgIdx, dx, dy, hA);

        // move original data from host to device
        if (CMDparams["type"] == "device")
        {
            dA = hA;
            dp = hp;
            db = hb;
        }
        world.barrier();
    }
    

    // initialize AmgX
    AMGX_SAFE_CALL(AMGX_initialize());
    world.barrier();

    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    world.barrier();


    // let AmgX use a user-defined output function to print messages
    if (myRank == 0) 
    { AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback)); }
    else 
    { AMGX_SAFE_CALL(AMGX_register_print_callback(&print_none)); }
    world.barrier();

    // let AmgX instead of users to handle messages returned by AmgX functions
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
    world.barrier();


    // create a config object using an input file
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, CMDparams["configFile"].c_str()));
    world.barrier();

    // to let the AmgX handle errors internally, so AMGX_SAFE_CALL can be 
    // discarded after this point and before the destroy of this config object
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));
    world.barrier();


    // create a resource object based on the current configuration
    MPI_Comm        comm = (ompi_communicator_t *)world;
    AMGX_resources_create(&rsrc, cfg, &comm, 1, devs.data());
    world.barrier();


    // set up the mode based on the input CMD parameter
    setMode(CMDparams["Mode"], mode);
    world.barrier();


    // assign memory space to the two vectors
    AMGX_vector_create(&AmgX_p, rsrc, mode);
    world.barrier();
    AMGX_vector_create(&AmgX_b, rsrc, mode);
    world.barrier();


    // assign memory space and an instance to the matrix variable "A"
    AMGX_matrix_create(&AmgX_A, rsrc, mode);
    world.barrier();

    
    // assign memory space and an instance to the solver variable "solver"
    // Note: after the solver is created, and before it is destroyed, 
    // the content of resource and config objects can not be modified!!
    AMGX_solver_create(&solver, rsrc, mode, cfg);
    world.barrier();


    int         rings;

    // obtain the default rings based on the config
    AMGX_config_get_default_number_of_rings(cfg, &rings);
    world.barrier();

    out << "Default rings: " << rings << std::endl;


    if (CMDparams.count("input"))
    {
        // read linear system (1 matrix, 2 vectors) from a .mtx file
        // AmgX will automatically read data distributedly
        world.barrier();
        AMGX_read_system_distributed(AmgX_A, AmgX_b, AmgX_p, 
                                     CMDparams["input"].c_str(), rings, 
                                     Npart, partSize.data(), 
                                     NpartVec, partVec.data());
        world.barrier();
        
        // check and show information of the linear systems
        rings = showSysInfo(AmgX_A, AmgX_b, AmgX_p);
        std::cout << rings << std::endl;
        world.barrier();
    }
    else
    {
        // pin host memory if the data are originally located on host
        if (CMDparams["type"] == "host")
        {
            b = hb.data();
            p = hp.data();
            Arow = hA.rowIdx.data();
            Acol = hA.colIdx.data();
            Adata = hA.data.data();

            AMGX_pin_memory(b, sizeof(double)*hb.size());
            AMGX_pin_memory(p, sizeof(double)*hp.size());
            AMGX_pin_memory(Arow, sizeof(int)*hA.rowIdx.size());
            AMGX_pin_memory(Acol, sizeof(long)*hA.colIdx.size());
            AMGX_pin_memory(Adata, sizeof(double)*hA.data.size());
        }
        else if (CMDparams["type"] == "device")
        {
            b = thrust::raw_pointer_cast(db.data());
            p = thrust::raw_pointer_cast(dp.data());
            Arow = thrust::raw_pointer_cast(dA.rowIdx.data());
            Acol = thrust::raw_pointer_cast(dA.colIdx.data());
            Adata = thrust::raw_pointer_cast(dA.data.data());

        }
        else
        {
            std::cerr << "Unknown type: " 
                      << CMDparams["type"] << std::endl;
            exit(0);
        }


        world.barrier();
        AMGX_matrix_upload_all_global(AmgX_A, Ntol, hA.Nrows, hA.Nnz, 1, 1, 
                               Arow, Acol, Adata, NULL, 
                               rings, rings, partVec.data());

        // copy data from original data to AmgX data structure
        timer.start();
        world.barrier();
        AMGX_vector_bind(AmgX_b, AmgX_A);

        world.barrier();
        AMGX_vector_bind(AmgX_p, AmgX_A);

        world.barrier();
        AMGX_vector_upload(AmgX_b, hA.Nrows, 1, b);

        world.barrier();
        AMGX_vector_upload(AmgX_p, hA.Nrows, 1, p);
        uploadTime = timer.format();
    }


    // bind A to the solver
    timer.start();
    AMGX_solver_setup(solver, AmgX_A);
    setupTime = timer.format();
    world.barrier();


    // solve
    timer.start();
    AMGX_solver_solve(solver, AmgX_b, AmgX_p);
    solveTime = timer.format();
    world.barrier();


    // get the status of the solver
    AMGX_solver_get_status(solver, &status);
    out << "Status: " << status << std::endl;


    // Download data from device to host
    timer.start();
    AMGX_vector_download(AmgX_p, p);
    downloadTime = timer.format();

    if (CMDparams["type"] == "device") hp = dp;
    world.barrier();


    // write A, b, p to an output .mtx file
    if (CMDparams.count("outputSys"))
        AMGX_write_system(AmgX_A, AmgX_b, AmgX_p, CMDparams["outputSys"].c_str());
    world.barrier();


    // write only p to a file
    if (CMDparams.count("outputResult"))
    {
        std::ofstream   file;
        for(int rank=0; rank<Npart; ++rank)
        {
            if (myRank == rank)
            {
                file.open(CMDparams["outputResult"], 
                        (myRank==0) ? std::ofstream::out : std::ofstream::app);
                for(auto it: hp) file << it << " ";
            }
            file.close();
            world.barrier();
        }
    }
    

    // destroy and finalize
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(AmgX_A);
    AMGX_vector_destroy(AmgX_b);
    AMGX_vector_destroy(AmgX_p);
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());


    out << std::endl;
    out << "Upload time: " << std::endl;
    out << "\t" << uploadTime << std::endl;
    out << "Setup time: " << std::endl;
    out << "\t" << setupTime << std::endl;
    out << "Solve time: " << std::endl;
    out << "\t" << solveTime << std::endl;
    out << "Download time: " << std::endl;
    out << "\t" << downloadTime << std::endl;

    return 0;
}


