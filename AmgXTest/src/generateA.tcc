
template<typename T>
void generateA(const int &Nx, const int &Ny, 
               const int &Nlocal, const int &bgIdx,
               const double &dx, const double &dy, SpMt<HOST, T> &A)
{
    double      cx = 1.0 / (dx * dx),
                cy = 1.0 / (dy * dy),
                cd = - 2.0 * (cx + cy);

    int         tempI = 0;

    A.Nnz = 0;
    A.Nrows = Nlocal;
    A.rowIdx.reserve(A.Nrows+1);
    A.colIdx.reserve(5 * A.Nrows);
    A.data.reserve(5 * A.Nrows);

    for(int i=0; i<A.Nrows; ++i)
    {
        T       col = i + bgIdx;
        int     grid_i = col % Nx,
                grid_j = col / Nx;

        A.rowIdx.push_back(A.Nnz);

        if (grid_i == 0 && grid_j == 0)
        {
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+1);
            A.colIdx.push_back(col+Nx);

            A.data.push_back(cd+cx+cy+1);
            A.data.push_back(cx);
            A.data.push_back(cy);

            A.Nnz += 3;
        }
        else if (grid_i == Nx - 1 && grid_j == 0)
        {
            A.colIdx.push_back(col-1);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+Nx);

            A.data.push_back(cx);
            A.data.push_back(cd+cx+cy);
            A.data.push_back(cy);

            A.Nnz += 3;
        }
        else if (grid_i == 0 && grid_j == Ny - 1)
        {
            A.colIdx.push_back(col-Nx);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+1);

            A.data.push_back(cy);
            A.data.push_back(cd+cx+cy);
            A.data.push_back(cx);

            A.Nnz += 3;
        }
        else if (grid_i == Nx - 1 && grid_j == Ny - 1)
        {
            A.colIdx.push_back(col-Nx);
            A.colIdx.push_back(col-1);
            A.colIdx.push_back(col);

            A.data.push_back(cy);
            A.data.push_back(cx);
            A.data.push_back(cd+cx+cy);

            A.Nnz += 3;
        }
        else if (grid_i == 0)
        {
            A.colIdx.push_back(col-Nx);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+1);
            A.colIdx.push_back(col+Nx);

            A.data.push_back(cy);
            A.data.push_back(cd+cx);
            A.data.push_back(cx);
            A.data.push_back(cy);

            A.Nnz += 4;
        }
        else if (grid_i == Nx - 1)
        {
            A.colIdx.push_back(col-Nx);
            A.colIdx.push_back(col-1);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+Nx);

            A.data.push_back(cy);
            A.data.push_back(cx);
            A.data.push_back(cd+cx);
            A.data.push_back(cy);

            A.Nnz += 4;
        }
        else if (grid_j == 0)
        {
            A.colIdx.push_back(col-1);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+1);
            A.colIdx.push_back(col+Nx);

            A.data.push_back(cx);
            A.data.push_back(cd+cy);
            A.data.push_back(cx);
            A.data.push_back(cy);

            A.Nnz += 4;
        }
        else if (grid_j == Ny - 1)
        {
            A.colIdx.push_back(col-Nx);
            A.colIdx.push_back(col-1);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+1);

            A.data.push_back(cy);
            A.data.push_back(cx);
            A.data.push_back(cd+cy);
            A.data.push_back(cx);

            A.Nnz += 4;
        }
        else
        {
            A.colIdx.push_back(col-Nx);
            A.colIdx.push_back(col-1);
            A.colIdx.push_back(col);
            A.colIdx.push_back(col+1);
            A.colIdx.push_back(col+Nx);

            A.data.push_back(cy);
            A.data.push_back(cx);
            A.data.push_back(cd);
            A.data.push_back(cx);
            A.data.push_back(cy);

            A.Nnz += 5;
        }
    }

    A.rowIdx.push_back(A.Nnz);
}
