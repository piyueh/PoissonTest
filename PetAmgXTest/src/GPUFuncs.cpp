# include <iostream>
# include <cuda_runtime.h>
# include "cudaCHECK.hpp"


int getMemUsage(const int &myRank)
{
    size_t free_byte,
           total_byte;

    CHECK(cudaMemGetInfo(&free_byte, &total_byte));

    std::cout << "myRank: " << myRank << " "
              << free_byte / 1024.0 / 1024.0 
              << " / " << total_byte / 1024.0 / 1024.0 << std::endl;

    return 0;
}
