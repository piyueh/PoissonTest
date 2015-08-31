# include "headers.hpp"


int generateGrid(const int &Nx, const int &Ny, 
        PetscScalar &dx, PetscScalar &dy, Vec &x, Vec &y)
{
    PetscErrorCode      ierr;   // error codes returned by PETSc routines

    dx = 1.0 / Nx;
    dy = 1.0 / Ny;

    for(int i=0; i<Nx; ++i)
    {
        VecSetValue(x, i, (i+0.5)*dx, INSERT_VALUES);             CHKERRQ(ierr);
    }
    VecAssemblyBegin(x); VecAssemblyEnd(x);

    for(int i=0; i<Ny; ++i)
    {
        VecSetValue(y, i, (i+0.5)*dy, INSERT_VALUES);             CHKERRQ(ierr);
    }
    VecAssemblyBegin(y); VecAssemblyEnd(y);

    return 0;
}

