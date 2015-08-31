# include "headers.hpp"


int generateRHS(const int &Nx, const int &Ny, 
        const Vec &x, const Vec &y, Vec &b)
{
    PetscInt            iBg, 
                        iEd;
    PetscScalar         v;
    PetscScalar         *raw_x, 
                        *raw_y;
    PetscErrorCode      ierr;   // error codes returned by PETSc routines

    ierr = VecGetArray(x, &raw_x);                                CHKERRQ(ierr);
    ierr = VecGetArray(y, &raw_y);                                CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(b, &iBg, &iEd);                   CHKERRQ(ierr);

    for(int idx=iBg; idx<iEd; ++idx)
    {
        int             grid_i = idx % Nx,
                        grid_j = idx / Nx;
        
        v = c2 * std::cos(c1 * raw_x[grid_i]) * std::cos(c1 * raw_y[grid_j]);
        ierr = VecSetValue(b, idx, v, ADD_VALUES);                CHKERRQ(ierr);
    }


    if (iBg == 0)
    {
        v = std::cos(c1 * raw_x[0]) * std::cos(c1 * raw_y[0]);
        ierr = VecSetValue(b, 0, v, ADD_VALUES);                  CHKERRQ(ierr);
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);                         CHKERRQ(ierr);

    ierr = VecAssemblyBegin(b);                                   CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);                                     CHKERRQ(ierr);

    ierr = VecRestoreArray(x, &raw_x);                            CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &raw_y);                            CHKERRQ(ierr);

    return 0;
}

