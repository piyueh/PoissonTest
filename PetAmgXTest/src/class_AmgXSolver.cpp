# include "class_AmgXSolver.hpp"


int AmgXSolver::count = 0;


int AmgXSolver::finalize()
{

    std::cout << 1 << std::endl;
    AMGX_solver_destroy(solver);
    std::cout << 2 << std::endl;
    AMGX_matrix_destroy(AmgXA);
    std::cout << 3 << std::endl;
    AMGX_vector_destroy(AmgXP);
    std::cout << 4 << std::endl;
    AMGX_vector_destroy(AmgXRHS);
    std::cout << 5 << std::endl;
    AMGX_resources_destroy(rsrc);
    std::cout << 6 << std::endl;
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    isInitialized = false;
    isUploaded_A = false;

    delete [] devs;

    if (count == 1)
    {
        std::cout << 7 << std::endl;
        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        std::cout << 8 << std::endl;
        AMGX_SAFE_CALL(AMGX_finalize());
    }

    count -= 1;

    return 0;
}


int AmgXSolver::initialize(MPI_Comm comm, int _Npart, int _myRank,
        const std::string &_mode, const std::string &cfg_file)
{
    count += 1;

    AmgXComm = comm;
    Npart = _Npart;
    myRank = _myRank;

    CHECK(cudaGetDeviceCount(&Ndevs));
    devs = new int(myRank % Ndevs);

    if (count == 1)
    {
        // initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        // intialize AmgX plugings
        AMGX_SAFE_CALL(AMGX_initialize_plugins());

        // use user defined output mechanism
        if (myRank == 0) 
        { 
            AMGX_SAFE_CALL(
                    AMGX_register_print_callback(&(AmgXSolver::print_callback))); 
        }
        else 
        { 
            AMGX_SAFE_CALL(
                    AMGX_register_print_callback(&(AmgXSolver::print_none))); 
        }

        // let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    // create an AmgX configure object
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfg_file.c_str()));

    // let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    // create an AmgX resource object
    AMGX_resources_create(&rsrc, cfg, &AmgXComm, 1, devs);

    // set mode
    setMode(_mode);

    // create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&AmgXP, rsrc, mode);
    AMGX_vector_create(&AmgXRHS, rsrc, mode);

    // create AmgX matrix object for unknowns and RHS
    AMGX_matrix_create(&AmgXA, rsrc, mode);

    // create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    // obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);

    isInitialized = true;
    
    return 0;
}


int AmgXSolver::setA(Mat &A)
{
    PetscInt        nLclRows,
                    nGlbRows;
    const PetscInt  *row, *col;
    Petsc64bitInt   *col64;
    PetscScalar     *data;
    PetscBool       done;
    Mat             lclA;
    PetscErrorCode  ierr;

    int            *partVec;

    // Get number of rows in global matrix
    ierr = MatGetSize(A, &nGlbRows, nullptr);                     CHKERRQ(ierr);

    // Get local matrix and its raw data
    ierr = MatMPIAIJGetLocalMat(A, MAT_INITIAL_MATRIX, &lclA);    CHKERRQ(ierr);
    ierr = MatGetRowIJ(lclA, 0, PETSC_FALSE, PETSC_FALSE, 
            &nLclRows, &row, &col, &done);                        CHKERRQ(ierr);
    ierr = MatSeqAIJGetArray(lclA, &data);                        CHKERRQ(ierr);

    // check whether MatGetRowIJ worked
    if (! done)
    {
        std::cerr << "MatGetRowIJ did not work!" << std::endl;
        exit(0);
    }

    // obtain a partition vector required for AmgX
    getPartVec(A, partVec);

    // cast 32bit integer array "col" to 64bit integer array "col64"
    col64 = new Petsc64bitInt[row[nLclRows]];
    for(size_t i=0; i<row[nLclRows]; ++i)
        col64[i] = static_cast<Petsc64bitInt>(col[i]);

    // upload matrix A to AmgX
    MPI_Barrier(AmgXComm);
    AMGX_matrix_upload_all_global(AmgXA, nGlbRows, nLclRows, row[nLclRows], 
            1, 1, row, col64, data, nullptr, ring, ring, partVec);

    // Return raw data to the local matrix (required by PETSc)
    ierr = MatRestoreRowIJ(lclA, 0, PETSC_FALSE, PETSC_FALSE, 
            &nLclRows, &row, &col, &done);                        CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArray(lclA, &data);                    CHKERRQ(ierr);

    // check whether MatRestoreRowIJ worked
    if (! done)
    {
        std::cerr << "MatRestoreRowIJ did not work!" << std::endl;
        exit(0);
    }

    // deallocate col64 and destroy local matrix
    delete [] col64;
    ierr = MatDestroy(&lclA);                                     CHKERRQ(ierr);

    // bind the matrix A to the solver
    MPI_Barrier(AmgXComm);
    AMGX_solver_setup(solver, AmgXA);

    // connect (bind) vectors to the matrix
    AMGX_vector_bind(AmgXP, AmgXA);
    AMGX_vector_bind(AmgXRHS, AmgXA);

    // change the state
    isUploaded_A = true;    

    MPI_Barrier(AmgXComm);

    return 0;
}


int AmgXSolver::solve(Vec &p, Vec &b)
{
    PetscErrorCode      ierr;
    double             *unks,
                       *rhs;
    int                 size;
    AMGX_SOLVE_STATUS   status;

    // get size of local vector (p and b should have the same local size)
    ierr = VecGetLocalSize(p, &size);                             CHKERRQ(ierr);

    // get pointers to the raw data of local vectors
    ierr = VecGetArray(p, &unks);                                 CHKERRQ(ierr);
    ierr = VecGetArray(b, &rhs);                                  CHKERRQ(ierr);

    // upload vectors to AmgX
    AMGX_vector_upload(AmgXP, size, 1, unks);
    AMGX_vector_upload(AmgXRHS, size, 1, rhs);

    // solve
    MPI_Barrier(AmgXComm);
    AMGX_solver_solve(solver, AmgXRHS, AmgXP);

    // get the status of the solver
    AMGX_solver_get_status(solver, &status);

    // check whether the solver successfully solve the problem
    if (status != AMGX_SOLVE_SUCCESS)
        std::cout << "AmgX solver failed to solve the problem! "
                  << "Solver status: " << status << std::endl;

    // download data from device
    AMGX_vector_download(AmgXP, unks);

    // restore PETSc vectors
    ierr = VecRestoreArray(p, &unks);                             CHKERRQ(ierr);
    ierr = VecRestoreArray(b, &rhs);                              CHKERRQ(ierr);

    return 0;
}

int AmgXSolver::setMode(const std::string &_mode)
{
    if (_mode == "hDDI")
        mode = AMGX_mode_hDDI;
    else if (_mode == "hDFI")
        mode = AMGX_mode_hDFI;
    else if (_mode == "hFFI")
        mode = AMGX_mode_hFFI;
    else if (_mode == "dDDI")
        mode = AMGX_mode_dDDI;
    else if (_mode == "dDFI")
        mode = AMGX_mode_dDFI;
    else if (_mode == "dFFI")
        mode = AMGX_mode_dFFI;
    else
    {
        std::cerr << "error: " 
                  << _mode << " is not an available mode." << std::endl;
        exit(0);
    }
    return 0;
}


int AmgXSolver::getPartVec(const Mat &A, int *& partVec)
{
    PetscErrorCode      ierr;
    PetscInt            size,
                        bg,
                        ed;
    VecScatter          scatter;
    Vec                 tempMPI,
                        tempSEQ;

    PetscScalar        *tempPartVec;

    ierr = MatGetOwnershipRange(A, &bg, &ed);                     CHKERRQ(ierr);
    ierr = MatGetSize(A, &size, nullptr);                         CHKERRQ(ierr);


    ierr = VecCreate(AmgXComm, &tempMPI);                         CHKERRQ(ierr);
    ierr = VecSetSizes(tempMPI, PETSC_DECIDE, size);              CHKERRQ(ierr);
    ierr = VecSetType(tempMPI, VECMPI);                           CHKERRQ(ierr);

    for(PetscInt i=bg; i<ed; ++i)
    {
        ierr = VecSetValue(tempMPI, i, myRank, INSERT_VALUES);    CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempMPI);                             CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempMPI);                               CHKERRQ(ierr);

    ierr = VecScatterCreateToAll(tempMPI, &scatter, &tempSEQ);    CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter, tempMPI, tempSEQ, 
            INSERT_VALUES, SCATTER_FORWARD);                      CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, tempMPI, tempSEQ, 
            INSERT_VALUES, SCATTER_FORWARD);                      CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);                           CHKERRQ(ierr);
    ierr = VecDestroy(&tempMPI);                                  CHKERRQ(ierr);

    ierr = VecGetArray(tempSEQ, &tempPartVec);                    CHKERRQ(ierr);

    partVec = new int[size];
    for(size_t i=0; i<size; ++i)
        partVec[i] = static_cast<int>(tempPartVec[i]);

    ierr = VecRestoreArray(tempSEQ, &tempPartVec);                CHKERRQ(ierr);

    ierr = VecDestroy(&tempSEQ);                                   CHKERRQ(ierr);
    ierr = VecDestroy(&tempMPI);                                   CHKERRQ(ierr);

    return 0;
}


void AmgXSolver::print_callback(const char *msg, int length)
{
    std::cout << msg;
}


void AmgXSolver::print_none(const char *msg, int length) { }
