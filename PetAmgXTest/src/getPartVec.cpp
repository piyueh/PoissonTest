# include "headers.hpp"

int getPartVec(const Vec &p, Vec &PartVec, int myRank)
{
    PetscErrorCode      ierr;
    PetscInt            size,
                        bg,
                        ed;
    VecScatter          scatter;
    Vec                 temp;

    ierr = VecGetOwnershipRange(p, &bg, &ed);                     CHKERRQ(ierr);
    ierr = VecGetSize(p, &size);                                  CHKERRQ(ierr);


    ierr = VecCreate(PETSC_COMM_WORLD, &temp);                    CHKERRQ(ierr);
    ierr = VecSetSizes(temp, PETSC_DECIDE, size);                 CHKERRQ(ierr);
    ierr = VecSetType(temp, VECMPI);                              CHKERRQ(ierr);

    for(PetscInt i=bg; i<ed; ++i)
    {
        ierr = VecSetValue(temp, i, myRank, INSERT_VALUES);       CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(temp);                                CHKERRQ(ierr);
    ierr = VecAssemblyEnd(temp);                                  CHKERRQ(ierr);

    ierr = VecScatterCreateToAll(temp, &scatter, &PartVec);       CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter, temp, PartVec, 
            INSERT_VALUES, SCATTER_FORWARD);                      CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, temp, PartVec, 
            INSERT_VALUES, SCATTER_FORWARD);                      CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);                           CHKERRQ(ierr);

    ierr = VecDestroy(&temp);                                     CHKERRQ(ierr);

    return 0;
}
