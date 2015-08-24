# include "headers.hpp"


void print_callback(const char *msg, int length)
{
    std::cout << msg;
}


void print_none(const char *msg, int length) { }


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
        ("outputSys,O", value<string>(), "Output MTX file for whole system")
        ("outputResult,o", value<string>(), "Output only result vector as txt file")
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

            if (vm.count("outputSys")) 
                CMDparams["outputSys"] = vm["outputSys"].as<string>();

            if (vm.count("outputResult")) 
                CMDparams["outputResult"] = vm["outputResult"].as<string>();

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
                 const int &Nlocal, const int &bgIdx,
                 HostVec &b, const HostVec &x, const HostVec &y)
{
    double      coeff1 = 2.0 * M_PI,
                coeff2 = - 8.0 * M_PI * M_PI;

    b.reserve(Nlocal);
    b.resize(Nlocal);

    for(int i=0; i<Nlocal; ++i)
    {
        int     grid_i = (i + bgIdx) % Nx,
                grid_j = (i + bgIdx) / Nx;

        b[i] = coeff2 * 
               std::cos(coeff1 * x[grid_i]) * 
               std::cos(coeff1 * y[grid_j]);
    }

    if (bgIdx == 0) // the gobally first node is located in rank 0
        b[0] += std::cos(coeff1 * x[0]) * std::cos(coeff1 * y[0]);
}


void generateZerosVec(const int &N, HostVec &p)
{
    p.reserve(N);
    p.resize(N);

    for(auto &it: p)    it = 0.0;
}


void exactSolution(const int &Nx, const int &Ny,
                   const int &Nlocal, const int &bgIdx,
                   HostVec &p_exact, const HostVec &x, const HostVec &y)
{
    double      coeff = 2.0 * M_PI;

    p_exact.reserve(Nlocal);
    p_exact.resize(Nlocal);

    for(int i=0; i<Nlocal; ++i)
    {
        int     grid_i = (i + bgIdx) % Nx,
                grid_j = (i + bgIdx) / Nx;

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

    exactSolution(Nx, Ny, Nx*Ny, 0, exact, x, y);

    L2 = errL2Norm(p.size(), p, exact);

    std::cout << "Error (L2 Norm): " << L2 << std::endl;
}


int fetchMtxSize(std::string mtxFile)
{
    std::ifstream       file(mtxFile);
    std::string         line;
    int                 N;

    while(std::getline(file, line))
    {
        std::istringstream      wholeLine(line);
        std::string             var;

        wholeLine >> var;

        if (var[0] != '%')
        {
            std::istringstream(var) >> N;
            break;
        }
    }

    return N;
}


void detPartSize(const int &Ntol, const int &mpiSize, const int &myRank,
                 int &Nlocal, int &bgIdx, int &edIdx, HostVecInt &partVec)
{
    int         Nbasic = Ntol / mpiSize,
                Nremain = Ntol % mpiSize;

    Nlocal = Nbasic;

    if (myRank < Nremain)
    {
        Nlocal = Nbasic + 1;
        bgIdx = (Nbasic + 1) * myRank;
    }
    else
        bgIdx = (Nbasic + 1) * Nremain + Nbasic * (myRank - Nremain);

    edIdx = bgIdx + Nlocal - 1;

    if ((myRank == mpiSize-1) && (edIdx != Ntol-1))
    {
        std::cerr << "error in the function detPartSize!" << std::endl;
        exit(0);
    }

    partVec.resize(Ntol);

    int     bg = 0, ed = 0;
    for(int rank=0; rank<mpiSize; ++rank)
    {
        if (rank < Nremain)
        {
            bg = (Nbasic + 1) * rank;
            ed = bg + Nbasic + 1 - 1;
        }
        else
        {
            bg = (Nbasic + 1) * Nremain + Nbasic * (rank - Nremain);
            ed = bg + Nbasic - 1;
        }

        for(int i=bg; i<=ed; ++i) partVec[i] = rank;
    }
}


template<typename T>
std::ostream & operator<<(std::ostream &os, std::vector<T> x)
{
	for(auto i: x) os << i << " ";
        os << std::endl;
	return os;
}
