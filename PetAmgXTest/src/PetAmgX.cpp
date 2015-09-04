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

    int                 size, myRank; // MPI size and current rank
                                      // devices used by current process

    char                mode[5],
                        file[128];

    AmgXSolver          solver;

    int                 event;

    // initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, nullptr, help.c_str());  CHKERRQ(ierr);

    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);                CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);              CHKERRQ(ierr);


    // get m and n from command-line argument
    ierr = PetscOptionsGetInt(nullptr, "-Nx", &Nx, nullptr);      CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(nullptr, "-Ny", &Ny, nullptr);      CHKERRQ(ierr);
    ierr = PetscOptionsGetString(nullptr, "-mode", mode, 5, nullptr);
                                                                  CHKERRQ(ierr);
    ierr = PetscOptionsGetString(nullptr, "-cfg", file, 128, nullptr);
                                                                  CHKERRQ(ierr);

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


    // initialize the AmgX solver instance
    solver.initialize(PETSC_COMM_WORLD, size, myRank, mode, file);

    // bind matrix A to the solver
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);
    solver.setA(A);


    ierr = PetscLogEventRegister("AmgX Section", 0, &event);      CHKERRQ(ierr);
    ierr = PetscLogEventBegin(event, 0, 0, 0, 0);                 CHKERRQ(ierr);

    // solve
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);
    solver.solve(p, b);

    ierr = PetscLogEventEnd(event, 0, 0, 0, 0);                 CHKERRQ(ierr);

    // destroy this instance and shutdown AmgX library
    solver.finalize();


    ierr = VecAXPY(p, -1, u);                                     CHKERRQ(ierr);
    ierr = VecNorm(p, NORM_2, &norm);                             CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "L2-Norm: %g\n", (double)norm);
                                                                  CHKERRQ(ierr);

    ierr = PetscFinalize();                                       CHKERRQ(ierr);

    return 0;
}

