# pragma once

# include <iostream>
# include <algorithm>
# include <string>
# include <vector>
# include <cmath>
# include <petscksp.h>

# define c1 2.0*1.0*M_PI
# define c2 -2.0*c1*c1


int generateGrid(const int &Nx, const int &Ny, 
        PetscScalar &dx, PetscScalar &dy, Vec &x, Vec &y);

int generateRHS(const int &Nx, const int &Ny, 
        const Vec &x, const Vec &y, Vec &b);

int generateExt(const int &Nx, const int &Ny, 
        const Vec &x, const Vec &y, Vec &u);

int generateA(const int &Nx, const int &Ny,
        const double &dx, const double &dy, Mat &A);

int getPartVec(const Vec &p, Vec &PartVec, int myRank);

int getMemUsage(const int &myRank);
