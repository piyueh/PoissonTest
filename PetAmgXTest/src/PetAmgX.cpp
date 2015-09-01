# include "headers.hpp"
# include "class_AmgXSolver.hpp"


static std::string help = "Test PETSc plus AmgX.";


int main(int argc, char **argv)
{
    PetscInt            Nx,     // number of elements in x-direction
                        Ny;     // number of elements in y-direction

    PetscScalar         dx,     // dx, calculated using Lx=1.0
                        dy;     // dy, calculated using Ly=1.0

    Vec                 x,      // x-coordinates
                        y,      // y-coordinates
                        p,      // unknowns
                        b,      // RHS
                        u;      // exact solution

    Mat                 A;      // coefficient matrix

    PetscReal           norm;   // norm of solution errors

    PetscErrorCode      ierr;   // error codes returned by PETSc routines

# ifdef PETSC_USE_LOG
    PetscLogStage       stage;
# endif

    int                 size, myRank; // MPI size and current rank
                                      // devices used by current process

    AmgXSolver          solver1, solver2;

    // initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, nullptr, help.c_str());  CHKERRQ(ierr);

    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);                CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);              CHKERRQ(ierr);


    // get m and n from command-line argument
    ierr = PetscOptionsGetInt(nullptr, "-Nx", &Nx, nullptr);      CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(nullptr, "-Ny", &Ny, nullptr);      CHKERRQ(ierr);


    // create vectors (x, y, p, b, u)
    ierr = VecCreate(PETSC_COMM_SELF, &x);                        CHKERRQ(ierr);
    ierr = VecSetSizes(x, Nx, Nx);                                CHKERRQ(ierr);
    ierr = VecSetType(x, VECSEQ);                                 CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_SELF, &y);                        CHKERRQ(ierr);
    ierr = VecSetSizes(y, Ny, Ny);                                CHKERRQ(ierr);
    ierr = VecSetType(y, VECSEQ);                                 CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD, &p);                       CHKERRQ(ierr);
    ierr = VecSetSizes(p, PETSC_DECIDE, Nx*Ny);                   CHKERRQ(ierr);
    ierr = VecSetType(p, VECMPI);                                 CHKERRQ(ierr);
    ierr = VecDuplicate(p, &b);                                   CHKERRQ(ierr);
    ierr = VecDuplicate(p, &u);                                   CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) x, "xCoor");          CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "yCoor");          CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) p, "Unknowns");       CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) b, "RHS");            CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Solution");       CHKERRQ(ierr);


    // set values of dx, dy, x, and y
    generateGrid(Nx, Ny, dx, dy, x, y);
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);


    // set values of RHS -- the vector b
    generateRHS(Nx, Ny, x, y, b);
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);


    // generate exact solution
    generateExt(Nx, Ny, x, y, u);
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);


    // set all entries as zeros in the vector of unknows
    ierr = VecSet(p, 0.0);                                        CHKERRQ(ierr);


    // initialize and set up the coefficient matrix
    ierr = MatCreate(PETSC_COMM_WORLD, &A);                       CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Nx*Ny, Nx*Ny); 
                                                                  CHKERRQ(ierr);
    ierr = MatSetType(A, MATMPIAIJ);                              CHKERRQ(ierr);
    ierr = MatSetUp(A);                                           CHKERRQ(ierr);

    generateA(Nx, Ny, dx, dy, A);
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);

    char *mode = "dDDI",
         *file = "configFiles/configMPI.amgx";

    solver1.initialize(PETSC_COMM_WORLD, size, myRank, mode, file);
    //solver2.initialize(PETSC_COMM_WORLD, size, myRank, mode, file);

    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);
    solver1.setA(A);
    //ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);
    //solver2.setA(A);

    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);
    solver1.solve(p, b);



    solver1.finalize();
    //solver2.finalize();

    ierr = PetscFinalize();                                       CHKERRQ(ierr);

    return 0;
}

