# pragma once

# include <iostream>
# include <algorithm>
# include <string>
# include <vector>
# include <cmath>
# include <cstring>

# include <petscsys.h>
# include <petscksp.h>
# include <petscdmda.h>

# include <boost/timer/timer.hpp>

# define c1 2.0*1.0*M_PI
# define c2 -3.0*c1*c1


PetscErrorCode generateGrid(const DM &grid, 
        const PetscInt &Nx, const PetscInt &Ny, const PetscInt &Nz,
        const PetscReal &Lx, const PetscReal &Ly, const PetscReal &Lz,
        PetscReal &dx, PetscReal &dy, PetscReal &dz,
        Vec &x, Vec &y, Vec &z);

PetscErrorCode generateRHS(const DM &grid, 
        const Vec &x, const Vec &y, const Vec &z, Vec &rhs);

PetscErrorCode generateExt(const DM &grid, 
        const Vec &x, const Vec &y, const Vec &z, Vec &exact);

PetscErrorCode generateA(const DM &grid, 
        const PetscReal &dx, const PetscReal &dy, const PetscReal &dz, Mat &A);

int getPartVec(const Vec &p, Vec &PartVec, int myRank);

int getMemUsage(const int &myRank);
