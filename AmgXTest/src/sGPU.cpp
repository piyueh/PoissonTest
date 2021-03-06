# include "headers.hpp"
# include "funcDef.hpp"


int main(int argc, char **argv)
{
    using namespace boost;

    // a map containning user-input parameters through CMD
    paramsMap       CMDparams;

    // parse CMD parameters and save into CMDparams
    parseCMD(argc, argv, CMDparams);

    // print out the user-input parameters
    std::cout << "Input parameters: " << std::endl;
    if (CMDparams.count("input"))
        std::cout << "\t" << "input: " << CMDparams["input"] << std::endl;
    else
    {
        for(auto it: CMDparams)
            std::cout << "\t" << it.first << ": " << it.second << std::endl;
    }


    // definition of original data
    // scalar parameters
    int                     Ntol,
                            Nx, 
                            Ny;
    double                  Lx,
                            Ly;
    double                  dx,
                            dy;


    // containers
    HostVec                     x,
                                y;
    HostSpMtInt                 hA;
    HostVec                     hp,
                                hb;
    DeviceSpMtInt               dA;
    DeviceVec                   dp,
                                db;

    // raw pointers
    double                      *b,
                                *p,
                                *Adata;
    int                         *Arow,
                                *Acol;

    // definition of all data related to AmgX
    AMGX_Mode               mode;
    AMGX_config_handle      cfg;
    AMGX_resources_handle   rsrc;
    AMGX_matrix_handle      AmgX_A;
    AMGX_vector_handle      AmgX_p,
                            AmgX_b;
    AMGX_solver_handle      solver;
    AMGX_SOLVE_STATUS       status;


    // variabled for timer
    timer::cpu_timer    timer;
    std::string         uploadTime,
                        setupTime,
                        solveTime,
                        downloadTime;



    if (! CMDparams.count("input"))
    {
        // initialize problem
        Nx = std::stoi(CMDparams["Nx"]);
        Ny = std::stoi(CMDparams["Ny"]);
        Ntol = Nx * Ny;

        Lx = 1.0;
        Ly = 1.0;

        generateXY(Nx, Ny, Lx, Ly, dx, dy, x, y);
        generateRHS(Nx, Ny, Ntol, 0, hb, x, y);
        generateZerosVec(Ntol, hp);
        generateA(Nx, Ny, Ntol, 0, dx, dy, hA);

        // move original data from host to device
        if (CMDparams["type"] == "device")
        {
            dA = hA;
            dp = hp;
            db = hb;
        }
    }
    

    // initialize AmgX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());


    // let AmgX use a user-defined output function to print messages
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    // let AmgX instead of users to handle messages returned by AmgX functions
    AMGX_SAFE_CALL(AMGX_install_signal_handler());


    // set up the mode based on the input CMD parameter
    setMode(CMDparams["Mode"], mode);


    // create a config object using an input file
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, 
                                CMDparams["configFile"].c_str()));
    // to let the AmgX handle errors internally, so AMGX_SAFE_CALL can be 
    // discarded after this point and before the destroy of this config object
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));


    // create a resource object based on the current configuration
    // "simple" means this is for single CPU thread and single GPU
    AMGX_resources_create_simple(&rsrc, cfg);


    // assign memory space to the two vectors
    AMGX_vector_create(&AmgX_p, rsrc, mode);
    AMGX_vector_create(&AmgX_b, rsrc, mode);


    // assign memory space and an instance to the matrix variable "A"
    AMGX_matrix_create(&AmgX_A, rsrc, mode);

    
    // assign memory space and an instance to the solver variable "solver"
    // Note: after the solver is created, and before it is destroyed, 
    // the content of resource and config objects can not be modified!!
    AMGX_solver_create(&solver, rsrc, mode, cfg);


    if (! CMDparams.count("input"))
    {
        // pin host memory
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
            AMGX_pin_memory(Acol, sizeof(int)*hA.colIdx.size());
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

        // copy data from original data to AmgX data structure
        timer.start();
        AMGX_vector_upload(AmgX_b, hA.Nrows, 1, b);
        AMGX_vector_upload(AmgX_p, hA.Nrows, 1, p);
        AMGX_matrix_upload_all(AmgX_A, hA.Nrows, hA.Nnz, 1, 1, 
                               Arow, Acol, Adata, nullptr);
        uploadTime = timer.format();
    }
    else
    {
        // read linear system (1 matrix, 2 vectors) from a .mtx file
        AMGX_read_system(AmgX_A, AmgX_b, AmgX_p, CMDparams["input"].c_str());
        
        // check and show information of the linear systems
        Ntol = showSysInfo(AmgX_A, AmgX_b, AmgX_p);
    }


    // bind A to the solver
    timer.start();
    AMGX_solver_setup(solver, AmgX_A);
    setupTime = timer.format();


    // solve
    timer.start();
    AMGX_solver_solve(solver, AmgX_b, AmgX_p);
    solveTime = timer.format();


    // get the status of the solver
    AMGX_solver_get_status(solver, &status);
    std::cout << "Status: " << status << std::endl;


    // Download data from device to host
    timer.start();
    AMGX_vector_download(AmgX_p, p);
    downloadTime = timer.format();

    if (CMDparams["type"] == "device") hp = dp;


    // write A, b, p to an output .mtx file
    if (CMDparams.count("outputSys"))
        AMGX_write_system(AmgX_A, AmgX_b, AmgX_p, CMDparams["outputSys"].c_str());


    // write only p to a file
    if (CMDparams.count("outputResult"))
    {
        std::ofstream   file(CMDparams["outputResult"]);
        for(auto it: hp) file << it << " ";
        file.close();
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


    std::cout << std::endl;
    std::cout << "Upload time: " << std::endl;
    std::cout << "\t" << uploadTime << std::endl;
    std::cout << "Setup time: " << std::endl;
    std::cout << "\t" << setupTime << std::endl;
    std::cout << "Solve time: " << std::endl;
    std::cout << "\t" << solveTime << std::endl;
    std::cout << "Download time: " << std::endl;
    std::cout << "\t" << downloadTime << std::endl;

    return 0;
}


