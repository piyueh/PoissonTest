# include "headers.hpp"

int generateA(const int &Nx, const int &Ny,
        const double &dx, const double &dy, Mat &A)
{
    PetscInt            iBg, 
                        iEd;
    PetscScalar         cx,
                        cy,
                        cdiag;
    PetscErrorCode      ierr;   // error codes returned by PETSc routines

    std::vector<PetscInt>       colIdx;
    std::vector<PetscScalar>    data;


    cx = 1.0 / dx / dx;
    cy = 1.0 / dy / dy;
    cdiag = -2.0 * (cx + cy);

    ierr = MatGetOwnershipRange(A, &iBg, &iEd);                   CHKERRQ(ierr);

    for(int iRow=iBg; iRow<iEd; ++iRow)
    {
        PetscInt        &col = iRow;
        PetscInt        grid_i = iRow % Nx,
                        grid_j = iRow / Nx;

        colIdx.clear(); data.clear();
        
        if (grid_i == 0 && grid_j == 0)
        {
            colIdx = {col, col+1, col+Nx};
            data = {cdiag+cx+cy+1, cx, cy};
        }
        else if (grid_i == Nx-1 && grid_j == 0)
        {
            colIdx = {col-1, col, col+Nx};
            data = {cx, cdiag+cx+cy, cy};
        }
        else if (grid_i == 0 && grid_j == Ny-1)
        {
            colIdx = {col-Nx, col, col+1};
            data = {cy, cdiag+cx+cy, cx};
        }
        else if (grid_i == Nx - 1 && grid_j == Ny-1)
        {
            colIdx = {col-Nx, col-1, col};
            data = {cy, cx, cdiag+cx+cy};
        }
        else if (grid_i == 0)
        {
            colIdx = {col-Nx, col, col+1, col+Nx};
            data = {cy, cdiag+cx, cx, cy};
        }
        else if (grid_i == Nx-1)
        {
            colIdx = {col-Nx, col-1, col, col+Nx};
            data = {cy, cx, cdiag+cx, cy};
        }
        else if (grid_j == 0)
        {
            colIdx = {col-1, col, col+1, col+Nx};
            data = {cx, cdiag+cy, cx, cy};
        }
        else if (grid_j == Ny-1)
        {
            colIdx = {col-Nx, col-1, col, col+1};
            data = {cy, cx, cdiag+cy, cx};
        }
        else
        {
            colIdx = {col-Nx, col-1, col, col+1, col+Nx};
            data = {cy, cx, cdiag, cx, cy};
        }

        ierr = MatSetValues(A, 1, &iRow, colIdx.size(), colIdx.data(), 
                            data.data(), INSERT_VALUES);          CHKERRQ(ierr);
    }


    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);               CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);                 CHKERRQ(ierr);

    return 0;
}
