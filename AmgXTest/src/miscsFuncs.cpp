# include "headers.hpp"


void print_callback(const char *msg, int length)
{
    std::cout << msg;
}


void parseCMD(int argc, char **argv, paramsMap & CMDparams)
{

    using namespace boost::program_options;
    using std::string;

    // database that describes all possible parameter
    options_description         helpArgList;
    options_description         mainArgList;
    options_description         allArgList("AmgX test.");

    // a map that contain parameters and corresponding values after parsing
    variables_map               vm;


    // set up the database
    helpArgList.add_options()
        ("help,h", "Print this help screen")
        ("version,v", "Print the version of AmgX");

    mainArgList.add_options()
        ("mode,m", value<string>()->default_value("dDDI"), "mode")
        ("configFile,c", value<string>()->required(), "Configuration file")
        ("input,i", value<string>(), "Input MatrixMarket file")
        ("output,o", value<string>(), "Output file")
        ("Nx", value<string>()->default_value("1000"), "Nx")
        ("Ny", value<string>()->default_value("1000"), "Ny")
        ("type,t", value<string>()->default_value("host"), 
                    "Whether to allocate original data on device or host.");

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
            CMDparams["type"] = vm["type"].as<string>();

            if (vm.count("output")) 
                CMDparams["output"] = vm["output"].as<string>();

            if (vm.count("input"))
            {
                CMDparams["input"] = vm["input"].as<string>();

                std::cout << "An input file has been specified, "
                          << "hence the settings of Nx, Ny, and type "
                          << "will be ignored." << std::endl;
            }
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
               const double &dx, const double &dy, HostSpMt &A)
{
    double      coeff_x = 1.0 / (dx * dx),
                coeff_y = 1.0 / (dy * dy),
                coeff_diag = - 2.0 * (coeff_x + coeff_y);

    int         tempI = 0;

    A.Nnz = 5 * Nx * Ny - 2 * Nx - 2 * Ny;
    A.Nrows = Nx * Ny;
    A.rowIdx.reserve(A.Nrows+1);
    A.colIdx.reserve(A.Nnz);
    A.data.reserve(A.Nnz);

    for(int i=0; i<A.Nrows; ++i)
    {
        int     grid_i = i % Nx,
                grid_j = i / Nx;

        A.rowIdx.push_back(tempI);

        if (grid_i == 0 && grid_j == 0)
        {
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+1);
            A.colIdx.push_back(i+Nx);

            A.data.push_back(coeff_diag + coeff_x + coeff_y + 1.0);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_y);

            tempI += 3;
        }
        else if (grid_i == Nx - 1 && grid_j == 0)
        {
            A.colIdx.push_back(i-1);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+Nx);

            A.data.push_back(coeff_x);
            A.data.push_back(coeff_diag + coeff_x + coeff_y);
            A.data.push_back(coeff_y);

            tempI += 3;
        }
        else if (grid_i == 0 && grid_j == Ny - 1)
        {
            A.colIdx.push_back(i-Nx);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+1);

            A.data.push_back(coeff_y);
            A.data.push_back(coeff_diag + coeff_x + coeff_y);
            A.data.push_back(coeff_x);

            tempI += 3;
        }
        else if (grid_i == Nx - 1 && grid_j == Ny - 1)
        {
            A.colIdx.push_back(i-Nx);
            A.colIdx.push_back(i-1);
            A.colIdx.push_back(i);

            A.data.push_back(coeff_y);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_diag + coeff_x + coeff_y);

            tempI += 3;
        }
        else if (grid_i == 0)
        {
            A.colIdx.push_back(i-Nx);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+1);
            A.colIdx.push_back(i+Nx);

            A.data.push_back(coeff_y);
            A.data.push_back(coeff_diag + coeff_x);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_y);

            tempI += 4;
        }
        else if (grid_i == Nx - 1)
        {
            A.colIdx.push_back(i-Nx);
            A.colIdx.push_back(i-1);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+Nx);

            A.data.push_back(coeff_y);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_diag + coeff_x);
            A.data.push_back(coeff_y);

            tempI += 4;
        }
        else if (grid_j == 0)
        {
            A.colIdx.push_back(i-1);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+1);
            A.colIdx.push_back(i+Nx);

            A.data.push_back(coeff_x);
            A.data.push_back(coeff_diag + coeff_y);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_y);

            tempI += 4;
        }
        else if (grid_j == Ny - 1)
        {
            A.colIdx.push_back(i-Nx);
            A.colIdx.push_back(i-1);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+1);

            A.data.push_back(coeff_y);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_diag + coeff_y);
            A.data.push_back(coeff_x);

            tempI += 4;
        }
        else
        {
            A.colIdx.push_back(i-Nx);
            A.colIdx.push_back(i-1);
            A.colIdx.push_back(i);
            A.colIdx.push_back(i+1);
            A.colIdx.push_back(i+Nx);

            A.data.push_back(coeff_y);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_diag);
            A.data.push_back(coeff_x);
            A.data.push_back(coeff_y);

            tempI += 5;
        }
    }


    if (tempI == A.Nnz)
        A.rowIdx.push_back(A.Nnz);
    else
    {
        std::cerr << "Wrong in generating matrix A!!" << std::endl;
        exit(0);
    }
}


void generateXY(const int &Nx, const int &Ny,
                const double &Lx, const double &Ly,
                double &dx, double &dy, HostVec &x, HostVec &y)
{
    dx = Lx / Nx;
    dy = Ly / Ny;

    x.reserve(Nx);
    y.reserve(Ny);

    for(int i=0; i<Nx; ++i) x.push_back(i*dx+dx/2.0);
    for(int j=0; j<Ny; ++j) y.push_back(j*dy+dy/2.0);
}


void generateRHS(const int &Nx, const int &Ny, 
                 HostVec &b, const HostVec &x, const HostVec &y)
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


void generateZerosVec(const int &N, HostVec &p)
{
    p.reserve(N);
    p.resize(N);

    for(auto &it: p)    it = 0.0;
}


void exactSolution(const int &Nx, const int &Ny,
                   HostVec &p_exact, const HostVec &x, const HostVec &y)
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


double errL2Norm(const int &N, const HostVec &p, const HostVec &p_exact)
{
    double      err = 0.0;

    for(size_t i=0; i<N; ++i)
        err += (p[i] - p_exact[i]) * (p[i] - p_exact[i]);

    err = std::sqrt(err);

    return err;
}


int showSysInfo(const AMGX_matrix_handle &A, 
                 const AMGX_vector_handle &b,
                 const AMGX_vector_handle &x)
{
    int         Nrows;

    // get the size information of A
    {
        int     bSizex, bSizey, nnz;

        AMGX_matrix_get_size(A, &Nrows, &bSizex, &bSizey);
        AMGX_matrix_get_nnz(A, &nnz);

        std::cout << "Matrix A: " << std::endl;
        if (bSizex == 1 && bSizey == 1)
            std::cout << "\tA is a scalar matrix." << std::endl;
        else
        {
            std::cout << "\tA is not a scalar matrix." << std::endl;
            std::cout << "\tBlock size of A: " 
                      << bSizex << " x " << bSizey << std::endl;
        }
        std::cout << "\tNumber of rows in A: " << Nrows << std::endl;
        std::cout << "\tNumber of non-zero entries: " << nnz << std::endl;
    }


    // get size information of b
    {
        int     Nvec, vecBsize;

        AMGX_vector_get_size(b, &Nvec, &vecBsize);

        std::cout << "RHS vector: " << std::endl;
        if (vecBsize == 1)
            std::cout << "\tRHS vector is a scalar vector." << std::endl;
        else
        {
            std::cout << "\tRHS vector is not a scalar vector." << std::endl;
            std::cout << "\tLength of a block is: " << vecBsize << std::endl;
        }
        std::cout << "\tNumber of entries in RHS vector: "
                  << Nvec << std::endl;
    }


    // get size information of x
    {
        int     Nvec, vecBsize;

        AMGX_vector_get_size(x, &Nvec, &vecBsize);

        std::cout << "Unknowns vector: " << std::endl;
        if (vecBsize == 1)
            std::cout << "\tUnknowns vector is a scalar vector." << std::endl;
        else
        {
            std::cout << "\tUnknowns vector is not a scalar vector." << std::endl;
            std::cout << "\tLength of a block is: " << vecBsize << std::endl;
        }
        std::cout << "\tNumber of entries in Unknowns vector: "
                  << Nvec << std::endl;
    }

    return Nrows;
}


void checkError(const int Nx, const int Ny, 
                const HostVec &p, const HostVec &x, const HostVec &y)
{
    double      L2;
    HostVec     exact(p.size());

    exactSolution(Nx, Ny, exact, x, y);

    L2 = errL2Norm(p.size(), p, exact);

    std::cout << "Error (L2 Norm): " << L2 << std::endl;
}


template<typename T>
std::ostream & operator<<(std::ostream &os, std::vector<T> x)
{
	for(auto i: x) os << i << " ";
        os << std::endl;
	return os;
}

