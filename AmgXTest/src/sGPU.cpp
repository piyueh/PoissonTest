# include <iostream>
# include <fstream>
# include <vector>
# include <string>
# include <map>
# include <cmath>
# include <amgx_c.h>
# include <boost/program_options.hpp>


typedef std::map<std::string, std::string>      paramsMap;
typedef std::vector<double>                     Vec;


class SpMt
{
    public:

        SpMt() = default;

        int                     Nrows,
                                Nnz;
        std::vector<int>        rowIdx,
                                colIdx;
        std::vector<double>     data;
};


void print_callback(const char *, int);
void init_CMDparser(int, char **, paramsMap &);
void setMode(std::string, AMGX_Mode &);

void generateA(const int &Nx, const int &Ny, 
               const double &dx, const double &dy, SpMt &A);

void generateXY(const int &Nx, const int &Ny,
                const double &Lx, const double &Ly,
                double &dx, double &dy, Vec &x, Vec &y);

void generateRHS(const int &Nx, const int &Ny, 
                 Vec &b, const Vec &x, const Vec &y);

void generateZerosVec(const int &N, Vec &p);

void exactSolution(const int &Nx, const int &Ny,
                   Vec &p_exact, const Vec &x, const Vec &y);
    
double errL2Norm(const int &N, const Vec &p, const Vec &p_exact);

template<typename T>
std::ostream & operator<<(std::ostream &os, std::vector<T> x);


int main(int argc, char **argv)
{

    paramsMap       CMDparams;

    init_CMDparser(argc, argv, CMDparams);

    for(auto it: CMDparams)
        std::cout << it.first << ": " << it.second << std::endl;


    AMGX_Mode               mode;
    AMGX_config_handle      cfg;
    AMGX_resources_handle   rsrc;
    AMGX_matrix_handle      AmgX_A;
    AMGX_vector_handle      AmgX_p,
                            AmgX_b;
    AMGX_solver_handle      solver;
    AMGX_SOLVE_STATUS       status;

    int                     Nx, 
                            Ny;
    double                  Lx,
                            Ly;
    double                  dx,
                            dy;


    SpMt                    A;
    Vec                     p,
                            b,
                            p_exact;
    Vec                     x,
                            y;

    Nx = std::stoi(CMDparams["Nx"]);
    Ny = std::stoi(CMDparams["Ny"]);

    Lx = 1.0;
    Ly = 1.0;

    generateXY(Nx, Ny, Lx, Ly, dx, dy, x, y);
    generateA(Nx, Ny, dx, dy, A);
    generateRHS(Nx, Ny, b, x, y);
    generateZerosVec(A.Nrows, p);
    exactSolution(Nx, Ny, p_exact, x, y);

    
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
    AMGX_vector_create(&AmgX_p, rsrc, mode);
    AMGX_vector_create(&AmgX_b, rsrc, mode);


    // assign memory space and an instance to the matrix variable "A"
    AMGX_matrix_create(&AmgX_A, rsrc, mode);

    
    // assign memory space and an instance to the solver variable "solver"
    // Note: after the solver is created, and before it is destroyed, 
    // the content of resource and config objects can not be modified!!
    AMGX_solver_create(&solver, rsrc, mode, cfg);


    // copy data from original data to AmgX data structure
    AMGX_vector_upload(AmgX_b, A.Nrows, 1, b.data());
    AMGX_vector_upload(AmgX_p, A.Nrows, 1, p.data());
    AMGX_matrix_upload_all(AmgX_A, A.Nrows, A.Nnz, 1, 1, 
            A.rowIdx.data(), A.colIdx.data(), A.data.data(), nullptr);


    // bind A to the solver
    AMGX_solver_setup(solver, AmgX_A);


    // solve
    AMGX_solver_solve(solver, AmgX_b, AmgX_p);


    // get the status of the solver
    AMGX_solver_get_status(solver, &status);
    std::cout << "Status: " << status << std::endl;

    // write A, b, x yo an output .mtx file
    if (CMDparams.count("output"))
        AMGX_write_system(AmgX_A, AmgX_b, AmgX_p, CMDparams["output"].c_str());
    

    // Download data from device to host
    AMGX_vector_download(AmgX_p, p.data());


    std::ofstream       solnFile("solution.txt");
    solnFile << p << std::endl;
    solnFile.close();


    double      err = errL2Norm(A.Nrows, p, p_exact);
    std::cout << "Error (L2 Norm): " << err << std::endl;


    // destroy and finalize
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(AmgX_A);
    AMGX_vector_destroy(AmgX_b);
    AMGX_vector_destroy(AmgX_p);
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
        ("configFile,c", value<string>()->required(), "Configuration file")
        ("output,o", value<string>(), "Output file")
        ("Nx", value<string>()->default_value("1000"), "Nx")
        ("Ny", value<string>()->default_value("1000"), "Ny");

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
            CMDparams["Nx"] = vm["Nx"].as<string>();
            CMDparams["Ny"] = vm["Ny"].as<string>();

            if (vm.count("output")) 
                CMDparams["output"] = vm["output"].as<string>();
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


void generateA(const int &Nx, const int &Ny, 
               const double &dx, const double &dy, SpMt &A)
{
    double      coeff_x = 1.0 / (dx * dx),
                coeff_y = 1.0 / (dy * dy),
                coeff_diag = - 2.0 * (coeff_x + coeff_y);

    //double      coeff_x = 1.0 * (dy * dy),
    //            coeff_y = 1.0 * (dx * dx),
    //            coeff_diag = - 2.0 * (coeff_x + coeff_y);

    //double      coeff_x = 1.0,
    //            coeff_y = 1.0,
    //            coeff_diag = - 4.0;

    A.Nnz = 0;
    A.Nrows = Nx * Ny;
    A.rowIdx.reserve(A.Nrows+1);
    A.colIdx.reserve(A.Nrows*5);
    A.data.reserve(A.Nrows*5);

    for(int i=0; i<A.Nrows; ++i)
    {
        int     grid_i = i % Nx,
                grid_j = i / Nx;

        A.rowIdx.emplace_back(A.Nnz);

        if (grid_i == 0 && grid_j == 0)
        {
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+1);
            A.colIdx.emplace_back(i+Nx);

            A.data.emplace_back(coeff_diag + coeff_x + coeff_y + 1.0);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_y);

            A.Nnz += 3;
        }
        else if (grid_i == Nx - 1 && grid_j == 0)
        {
            A.colIdx.emplace_back(i-1);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+Nx);

            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_diag + coeff_x + coeff_y);
            A.data.emplace_back(coeff_y);

            A.Nnz += 3;
        }
        else if (grid_i == 0 && grid_j == Ny - 1)
        {
            A.colIdx.emplace_back(i-Nx);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+1);

            A.data.emplace_back(coeff_y);
            A.data.emplace_back(coeff_diag + coeff_x + coeff_y);
            A.data.emplace_back(coeff_x);

            A.Nnz += 3;
        }
        else if (grid_i == Nx - 1 && grid_j == Ny - 1)
        {
            A.colIdx.emplace_back(i-Nx);
            A.colIdx.emplace_back(i-1);
            A.colIdx.emplace_back(i);

            A.data.emplace_back(coeff_y);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_diag + coeff_x + coeff_y);

            A.Nnz += 3;
        }
        else if (grid_i == 0)
        {
            A.colIdx.emplace_back(i-Nx);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+1);
            A.colIdx.emplace_back(i+Nx);

            A.data.emplace_back(coeff_y);
            A.data.emplace_back(coeff_diag + coeff_x);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_y);

            A.Nnz += 4;
        }
        else if (grid_i == Nx - 1)
        {
            A.colIdx.emplace_back(i-Nx);
            A.colIdx.emplace_back(i-1);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+Nx);

            A.data.emplace_back(coeff_y);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_diag + coeff_x);
            A.data.emplace_back(coeff_y);

            A.Nnz += 4;
        }
        else if (grid_j == 0)
        {
            A.colIdx.emplace_back(i-1);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+1);
            A.colIdx.emplace_back(i+Nx);

            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_diag + coeff_y);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_y);

            A.Nnz += 4;
        }
        else if (grid_j == Ny - 1)
        {
            A.colIdx.emplace_back(i-Nx);
            A.colIdx.emplace_back(i-1);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+1);

            A.data.emplace_back(coeff_y);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_diag + coeff_y);
            A.data.emplace_back(coeff_x);

            A.Nnz += 4;
        }
        else
        {
            A.colIdx.emplace_back(i-Nx);
            A.colIdx.emplace_back(i-1);
            A.colIdx.emplace_back(i);
            A.colIdx.emplace_back(i+1);
            A.colIdx.emplace_back(i+Nx);

            A.data.emplace_back(coeff_y);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_diag);
            A.data.emplace_back(coeff_x);
            A.data.emplace_back(coeff_y);

            A.Nnz += 5;
        }
    }

    A.rowIdx.emplace_back(A.Nnz);
    A.colIdx.shrink_to_fit();
    A.data.shrink_to_fit();
}


void generateXY(const int &Nx, const int &Ny,
                const double &Lx, const double &Ly,
                double &dx, double &dy, Vec &x, Vec &y)
{
    dx = Lx / Nx;
    dy = Ly / Ny;

    x.reserve(Nx);
    y.reserve(Ny);

    for(int i=0; i<Nx; ++i) x.emplace_back(i*dx+dx/2.0);
    for(int j=0; j<Ny; ++j) y.emplace_back(j*dy+dy/2.0);
}


void generateRHS(const int &Nx, const int &Ny, 
                 Vec &b, const Vec &x, const Vec &y)
{
    double      coeff1 = 2.0 * M_PI,
                coeff2 = - 8.0 * M_PI * M_PI;

    b.reserve(Nx*Ny);
    b.resize(Nx*Ny);

    for(int i=0; i<Nx*Ny; ++i)
    {
        int     grid_i = i % Nx,
                grid_j = i / Nx;

        b[i] = coeff2 * 
               std::cos(coeff1 * x[grid_i]) * 
               std::cos(coeff1 * y[grid_j]);
    }

    b[0] += std::cos(coeff1 * x[0]) * std::cos(coeff1 * y[0]);
}


void generateZerosVec(const int &N, Vec &p)
{
    p.reserve(N);
    p.resize(N);

    for(auto &it: p)    it = 0.0;
}


void exactSolution(const int &Nx, const int &Ny,
                   Vec &p_exact, const Vec &x, const Vec &y)
{
    double      coeff = 2.0 * M_PI;

    p_exact.reserve(Nx*Ny);
    p_exact.resize(Nx*Ny);

    for(int i=0; i<Nx*Ny; ++i)
    {
        int     grid_i = i % Nx,
                grid_j = i / Nx;

        p_exact[i] = std::cos(coeff * x[grid_i]) * 
                     std::cos(coeff * y[grid_j]);
    }
}


double errL2Norm(const int &N, const Vec &p, const Vec &p_exact)
{
    double      err = 0.0;

    for(size_t i=0; i<N; ++i)
        err += (p[i] - p_exact[i]) * (p[i] - p_exact[i]);

    err = std::sqrt(err);

    return err;
}


template<typename T>
std::ostream & operator<<(std::ostream &os, std::vector<T> x)
{
	for(auto i: x) os << i << " ";
	return os;
}
