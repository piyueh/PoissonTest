# include "headers.hpp"
# include "class_AmgXSolver.hpp"
# include <cuda_runtime.h>
# include "cudaCHECK.hpp"


static std::string help = "Test pure PETSc solvers.";


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

    PetscReal           norm2,
                        normM;  // norm of solution errors

    PetscInt            Niters; // iterations used to converge

    PetscErrorCode      ierr;   // error codes returned by PETSc routines

    int                 size, myRank; // MPI size and current rank
                                      // devices used by current process




    KSP                 ksp;    // krylov solver object

    PC                  pc;     // preconditioner

    // initialize PETSc
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


    // initialize and set the coefficient matrix
    ierr = MatCreate(PETSC_COMM_WORLD, &A);                       CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Nx*Ny, Nx*Ny); 
                                                                  CHKERRQ(ierr);
    ierr = MatSetType(A, MATMPIAIJ);                              CHKERRQ(ierr);
    ierr = MatSetUp(A);                                           CHKERRQ(ierr);

    generateA(Nx, Ny, dx, dy, A);
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);


    // initialize the PETSc solver instance
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);                     CHKERRQ(ierr);

    // if there's CMD parameters start with -poisson_, it's solver parameter
    ierr = KSPSetOptionsPrefix(ksp, "poisson_"); 		  CHKERRQ(ierr);

    // bind matrix A to the solver
    ierr = KSPSetOperators(ksp, A, A);                            CHKERRQ(ierr);

    // set default settings for the case that no solver is specified through CMD
    ierr = KSPSetType(ksp, KSPCG);                                CHKERRQ(ierr);
    ierr = KSPCGSetType(ksp, KSP_CG_SYMMETRIC);                   CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);            CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);        CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-08, PETSC_DEFAULT, PETSC_DEFAULT, 10000000);
                                                                  CHKERRQ(ierr);

    // now read parameters from CMD
    ierr = KSPSetFromOptions(ksp); 				  CHKERRQ(ierr);

    // solve
    ierr = KSPSolve(ksp, b, p);                                   CHKERRQ(ierr);

    // get number of iterations, it doesn't mean the solver converge
    ierr = KSPGetIterationNumber(ksp, &Niters);                   CHKERRQ(ierr);

    // calculate norms of errors
    ierr = VecAXPY(p, -1.0, u);                                   CHKERRQ(ierr);
    ierr = VecNorm(p, NORM_2, &norm2);                            CHKERRQ(ierr);
    ierr = VecNorm(p, NORM_INFINITY, &normM);                     CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "L2-Norm: %g\n", (double)norm2);
                                                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Max-Norm: %g\n", (double)normM);
                                                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations %D\n", Niters);                                CHKERRQ(ierr);
                                                                  CHKERRQ(ierr);

    // destroy KSP
    ierr = KSPDestroy(&ksp);                                      CHKERRQ(ierr);

    // finalize PETSc
    ierr = PetscFinalize();                                       CHKERRQ(ierr);

    return 0;
}

