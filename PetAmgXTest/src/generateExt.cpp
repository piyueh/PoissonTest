# include "headers.hpp"


int generateExt(const int &Nx, const int &Ny, 
        const Vec &x, const Vec &y, Vec &u)
{
    PetscInt            iBg, 
                        iEd;
    PetscScalar         v;
    PetscScalar         *raw_x, 
                        *raw_y;
    PetscErrorCode      ierr;   // error codes returned by PETSc routines

    ierr = VecGetArray(x, &raw_x);                                CHKERRQ(ierr);
    ierr = VecGetArray(y, &raw_y);                                CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(u, &iBg, &iEd);                   CHKERRQ(ierr);

    for(int idx=iBg; idx<iEd; ++idx)
    {
        int             grid_i = idx % Nx,
                        grid_j = idx / Nx;
        
        v = std::cos(c1 * raw_x[grid_i]) * std::cos(c1 * raw_y[grid_j]);
        ierr = VecSetValue(u, idx, v, ADD_VALUES);                CHKERRQ(ierr);
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);

    ierr = VecAssemblyBegin(u);                                   CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u);                                     CHKERRQ(ierr);

    ierr = VecRestoreArray(x, &raw_x);                            CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &raw_y);                            CHKERRQ(ierr);

    return 0;
}

