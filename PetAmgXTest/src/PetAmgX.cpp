# include "headers.hpp"
# include "AmgXSolver.hpp"
# include <cuda_runtime.h>
# include "cudaCHECK.hpp"

# define CHK CHKERRQ(ierr)
# define CHKMSG(flag, message)                  \
    if (!flag)                                  \
    {                                           \
        PetscPrintf(PETSC_COMM_WORLD, message); \
        PetscPrintf(PETSC_COMM_WORLD, "\n");    \
        exit(EXIT_FAILURE);                     \
    }

static std::string help = "Test PETSc plus AmgX solvers.";


int main(int argc, char **argv)
{
    using namespace boost;

    PetscInt            Nx,     // number of elements in x-direction
                        Ny,     // number of elements in y-direction
                        Nz;     // number of elements in z-direction

    PetscReal           Lx = 1.0, // Lx
                        Ly = 1.0, // Ly
                        Lz = 1.0; // Lz

    PetscReal           dx,     // dx, calculated using Lx=1.0
                        dy,     // dy, calculated using Ly=1.0
                        dz;     // dy, calculated using Ly=1.0

    DM                  grid;   // DM object

    Vec                 x,      // x-coordinates
                        y,      // y-coordinates
                        z;      // z-coordinates

    Vec                 u,      // unknowns
                        rhs,    // RHS
                        bc,     // boundary conditions
                        u_exact;// exact solution

    Mat                 A;      // coefficient matrix

    KSP                 ksp;

    AmgXSolver          amgx;

    PetscReal           norm2,
                        normM;   // norm of solution errors

    PetscInt            Niters; // iterations used to converge

    PetscErrorCode      ierr;   // error codes returned by PETSc routines

    PetscMPIInt         size,   // MPI size
                        myRank; // rank of current process

    PetscBool           set;

    KSPConvergedReason  reason;

    PetscViewer         viewer;


    std::string         solveTime;

    std::string         mode = "dDDI";

    char                cfgFileName[PETSC_MAX_PATH_LEN],
                        optFileName[PETSC_MAX_PATH_LEN],
                        platform[4],
                        caseName[PETSC_MAX_PATH_LEN];

    PetscLogStage       stageSolving;

    timer::cpu_timer    timer;


    // initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, nullptr, help.c_str());             CHK;
    ierr = PetscLogDefaultBegin();                                           CHK;
    ierr = PetscLogStageRegister("solving", &stageSolving);                  CHK;


    // obtain the rank and size of MPI
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);                           CHK;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);                         CHK;


    // get parameters from command-line arguments
    ierr = PetscOptionsGetString(nullptr, "-caseName", 
            caseName, PETSC_MAX_PATH_LEN, &set);                             CHK;
    CHKMSG(set, "caseName not yet set!");

    ierr = PetscOptionsGetInt(nullptr, "-Nx", &Nx, &set);                    CHK;
    CHKMSG(set, "Nx not yet set!");

    ierr = PetscOptionsGetInt(nullptr, "-Ny", &Ny, &set);                    CHK;
    CHKMSG(set, "Nx not yet set!");

    ierr = PetscOptionsGetInt(nullptr, "-Nz", &Nz, &set);                    CHK;
    CHKMSG(set, "Nx not yet set!");

    ierr = PetscOptionsGetString(nullptr, "-platform", platform, 4, &set);   CHK;
    CHKMSG(set, "platform not yet set!");

    ierr = PetscOptionsGetString(nullptr, "-cfgFileName", 
            cfgFileName, PETSC_MAX_PATH_LEN, &set);                          CHK;
    CHKMSG(set, "cfgFileName (configuration file) not yet set!");

    ierr = PetscOptionsGetString(nullptr, "-optFileName", 
            optFileName, PETSC_MAX_PATH_LEN, &set);                          CHK;
    CHKMSG(set, "optFileName (output file) not yet set!");


    // create DMDA object
    ierr = DMDACreate3d(PETSC_COMM_WORLD, 
            DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
            DMDA_STENCIL_STAR, 
            Nx, Ny, Nz,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            1, 1, nullptr, nullptr, nullptr, &grid);                         CHK;

    ierr = PetscObjectSetName((PetscObject) grid, "DMDA representing grid"); CHK;
            

    // create vectors (x, y, p, b, u)
    ierr = DMCreateGlobalVector(grid, &x);                                   CHK;
    ierr = DMCreateGlobalVector(grid, &y);                                   CHK;
    ierr = DMCreateGlobalVector(grid, &z);                                   CHK;
    ierr = DMCreateGlobalVector(grid, &u);                                   CHK;
    ierr = DMCreateGlobalVector(grid, &rhs);                                 CHK;
    ierr = DMCreateGlobalVector(grid, &bc);                                  CHK;
    ierr = DMCreateGlobalVector(grid, &u_exact);                             CHK;

    ierr = PetscObjectSetName((PetscObject) x, "x coordinates");             CHK;
    ierr = PetscObjectSetName((PetscObject) y, "y coordinates");             CHK;
    ierr = PetscObjectSetName((PetscObject) z, "z coordinates");             CHK;
    ierr = PetscObjectSetName((PetscObject) u, "vec for unknowns");          CHK;
    ierr = PetscObjectSetName((PetscObject) rhs, "RHS");                     CHK;
    ierr = PetscObjectSetName((PetscObject) bc, "boundary conditions");      CHK;
    ierr = PetscObjectSetName((PetscObject) u_exact, "exact solution");      CHK;


    // set values of dx, dy, dx, x, y, and z
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                                    CHK;
    ierr = generateGrid(grid, Nx, Ny, Nz, Lx, Ly, Lz, dx, dy, dz, x, y, z);  CHK;


    // set values of RHS -- the vector rhs
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                                    CHK;
    ierr = generateRHS(grid, x, y, z, rhs);                                  CHK;


    // generate exact solution
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                                    CHK;
    ierr = generateExt(grid, x, y, z, u_exact);                              CHK;


    // set all entries as zeros in the vector of unknows
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                                    CHK;
    ierr = VecSet(u, 0.0);                                                   CHK;


    // initialize and set up the coefficient matrix
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                                    CHK;
    ierr = DMSetMatType(grid, MATAIJ);                                       CHK;
    ierr = DMCreateMatrix(grid, &A);                                         CHK;
    ierr = MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);        CHK;
    ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);      CHK;

    ierr = generateA(grid, dx, dy, dz, A);                                   CHK;

    if (std::strcmp(platform, "CPU") == 0)
    {
        ierr = createKSP(ksp, A, cfgFileName);                               CHK;

        ierr = PetscLogStagePush(stageSolving);                              CHK;

        timer.start();
        ierr = KSPSolve(ksp, rhs, u);                                        CHK;
        solveTime = timer.format();

        ierr = PetscLogStagePop();                                           CHK;

        ierr = KSPGetConvergedReason(ksp, &reason);                          CHK;
        if (reason < 0)
        {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "\nDiverged: %d\n", reason);CHK;
            exit(0);
        }

        ierr = KSPGetIterationNumber(ksp, &Niters);                          CHK;
    }
    else if (std::strcmp(platform, "GPU") == 0)
    {
        amgx.initialize(PETSC_COMM_WORLD, size, myRank, "dDDI", cfgFileName);

        ierr = MPI_Barrier(PETSC_COMM_WORLD);                                CHK;
        amgx.setA(A);

        ierr = MPI_Barrier(PETSC_COMM_WORLD);                                CHK;
        ierr = PetscLogStagePush(stageSolving); CHK;

        timer.start();
        amgx.solve(u, rhs);
        solveTime = timer.format();

        ierr = PetscLogStagePop();

        Niters = amgx.getIters();
    }

    // calculate norms of errors
    ierr = VecAXPY(u, -1.0, u_exact);                                        CHK;
    ierr = VecNorm(u, NORM_2, &norm2);                                       CHK;
    ierr = VecNorm(u, NORM_INFINITY, &normM);                                CHK;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "L2-Norm: %g\n", (double)norm2);    CHK;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Max-Norm: %g\n", (double)normM);   CHK;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Iterations %D\n", Niters);         CHK; 


    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, optFileName, &viewer); CHK;
    ierr = PetscLogView(viewer); CHK;
    ierr = PetscViewerDestroy(&viewer); CHK;
    

    if (std::strcmp(platform, "CPU") == 0)
        ierr = KSPDestroy(&ksp);
    else if (std::strcmp(platform, "GPU") == 0)
        amgx.finalize();


    ierr = PetscPrintf(PETSC_COMM_WORLD, "Case Name: %s; Solve Time: %s\n", 
            caseName, solveTime.c_str());                                    CHK;

    // finalize PETSc
    ierr = PetscFinalize();                                                  CHK;

    return 0;
}

