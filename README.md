Poisson Test
====================

This repository contains poisson test cases using different linear algebra packages in order to test the capacities of different libraries.

- PetAmgXTest: use PETSc as the distributed linear algebra library and AmgX as the multi-GPU solver. A wrapper for AmgX to be used in PETSc is included in. 

- AmgXTest: use AmgX as the solver. No other linear algebra libraries are used. Distributed matrices and vectors are coded manually.  The codes are not well orginized because they are just for some rough tests.

- TrilinosTest: use Trilinos to test its multi-GPU capacity. These codes haven't been maintained for a while. It's a mess.
