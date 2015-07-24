# include <iostream>
# include <string>
# include <map>
# include <amgx_c.h>
# include <boost/program_options.hpp>


typedef std::map<std::string, std::string>      paramsMap;


void print_callback(const char *, int);
void init_CMDparser(int, char **, paramsMap &);
void setMode(std::string, AMGX_Mode &);


int main(int argc, char **argv)
{

    paramsMap       CMDparams;

    init_CMDparser(argc, argv, CMDparams);

    for(auto it: CMDparams)
        std::cout << it.first << ": " << it.second << std::endl;


    AMGX_Mode               mode;
    AMGX_config_handle      cfg;
    AMGX_resources_handle   rsrc;
    AMGX_matrix_handle      A;
    AMGX_vector_handle      x,
                            b;
    AMGX_solver_handle      solver;
    AMGX_SOLVE_STATUS       status;

    int                     Nmtx, 
                            bSizex, 
                            bSizey, 
                            nnz,
                            Nvec, 
                            vecBsize;

    // initialize
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
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);


    // assign memory space and an instance to the matrix variable "A"
    AMGX_matrix_create(&A, rsrc, mode);

    
    // assign memory space and an instance to the solver variable "solver"
    // Note: after the solver is created, and before it is destroyed, 
    // the content of resource and config objects can not be modified!!
    AMGX_solver_create(&solver, rsrc, mode, cfg);


    // read linear system (1 matrix, 2 vectors) from a .mtx file
    AMGX_read_system(A, b, x, CMDparams["mtxFile"].c_str());
    // get the size information of A
    AMGX_matrix_get_size(A, &Nmtx, &bSizex, &bSizey);
    std::cout << Nmtx << " " << bSizex << " " << bSizey << std::endl;
    // get non-zero entries of A
    AMGX_matrix_get_nnz(A, &nnz);
    std::cout << nnz << std::endl;
    // get size information of b
    AMGX_vector_get_size(b, &Nvec, &vecBsize);
    std::cout << Nvec << " " << vecBsize << std::endl;
    // get size information of f
    AMGX_vector_get_size(x, &Nvec, &vecBsize);
    std::cout << Nvec << " " << vecBsize << std::endl;


    // bind A to the solver
    AMGX_solver_setup(solver, A);


    // solve
    AMGX_solver_solve(solver, b, x);


    // get the status of the solver
    AMGX_solver_get_status(solver, &status);
    std::cout << "Status: " << status << std::endl;

    // write A, b, x yo an output .mtx file
    AMGX_write_system(A, b, x, "output.mtx");


    // destroy and finalize
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(A);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(x);
    AMGX_resources_destroy(rsrc);

    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());

    return 0;
}


void print_callback(const char *msg, int length)
{
    std::cout << msg;
}


void init_CMDparser(int argc, char **argv, paramsMap & CMDparams)
{

    using namespace boost::program_options;
    using std::string;

    // database that describes all possible parameter
    options_description         helpArgList;
    options_description         mainArgList;
    options_description         allArgList("AmgX test.");

    // a map that contain parameters and corresponding values after parsing
    variables_map          vm;


    // set up the database
    helpArgList.add_options()
        ("help,h", "Print this help screen")
        ("version,v", "Print the version of AmgX");

    mainArgList.add_options()
        ("mode,m", value<string>()->default_value("dDDI"), "mode")
        ("configFile,c", value<string>()->required(), 
                                        "Name of the configuration file")
        ("mtxFile,M", value<string>()->required(),
                                        "Name of the MatrixMarket file");

    allArgList.add(helpArgList);
    allArgList.add(mainArgList);


    try
    {
        // parse_command_line parses the command line arguments and returns
        // an object of type "parsed-options"
        // store() put the content of an object of type "parsed-object" into
        // a map (variable_map is derived from STL map)
        command_line_parser     helpParser(argc, argv);
        helpParser.options(helpArgList);
        helpParser.allow_unregistered();
        store(helpParser.run(), vm);

        // notify() activates all parameters that have "notifier". It also saves
        // values into corresponding variables if there are variables in 
        // value<>().
        notify(vm);

        if (vm.count("help"))
        {
            std::cout << allArgList << std::endl;
            exit(0);
        }
        else if (vm.count("version"))
        {
            int     major, minor;
            char    *ver, *date, *time;

            AMGX_get_api_version(&major, &minor);
            AMGX_get_build_info_strings(&ver, &date, &time);

            std::cout << "AmgX api version: " 
                      << major << "." << minor << std::endl;
            std::cout << "AmgX build version: "
                      << ver << std::endl
                      << "Build date and time: " << date << " " << time 
                      << std::endl;
            exit(0);    
        }
        else
        {
            // another way to parse arguments
            store(parse_command_line(argc, argv, mainArgList), vm);
            notify(vm);

            CMDparams["Mode"] = vm["mode"].as<string>();
            CMDparams["configFile"] = vm["configFile"].as<string>();
            CMDparams["mtxFile"] = vm["mtxFile"].as<string>();
        }
    }
    catch(const error &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
}


void setMode(std::string mode_, AMGX_Mode & mode)
{
    if (mode_ == "hDDI")
        mode = AMGX_mode_hDDI;
    else if (mode_ == "hDFI")
        mode = AMGX_mode_hDFI;
    else if (mode_ == "hFFI")
        mode = AMGX_mode_hFFI;
    else if (mode_ == "dDDI")
        mode = AMGX_mode_dDDI;
    else if (mode_ == "dDFI")
        mode = AMGX_mode_dDFI;
    else if (mode_ == "dFFI")
        mode = AMGX_mode_dFFI;
    else
    {
        std::cerr << "error: " 
                  << mode_ << " is not an available mode." << std::endl;
        exit(0);
    }
}
