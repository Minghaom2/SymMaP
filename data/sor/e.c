#include <petsc.h>
#include <stdio.h>

int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, NULL, NULL);

    int nnz, n;
    PetscInt its;
    int *row, *col;
    double *data;
    PetscLogDouble start, finish;
    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;
    PetscViewer viewer;

    FILE *file = fopen("A.bin", "rb");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    fread(&nnz, sizeof(int), 1, file);
    fread(&n, sizeof(int), 1, file);
    row = (int *) malloc (nnz * sizeof(int));
    col = (int *) malloc (nnz * sizeof(int));
    data = (double *) malloc (nnz * sizeof(double));
    for (int i=0;i<nnz;i++){
        fread(row+i, sizeof(int), 1, file);
    }
    for (int i=0;i<nnz;i++){
        fread(col+i, sizeof(int), 1, file);
    }
    for (int i=0;i<nnz;i++){
        fread(data+i, sizeof(double), 1, file);
    }

    fclose(file);

    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATAIJ));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
    PetscCall(MatSetUp(A));
    for (int i=0;i<nnz;i++) {
        PetscCall(MatSetValue(A, row[i], col[i], data[i], INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
    PetscCall(VecSetSizes(b, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(b));
    PetscCall(VecSetUp(b));
    PetscCall(VecSetRandom(b, NULL));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetUp(ksp));
    PetscCall(VecDuplicate(b, &x));

    PetscCall(KSPGetPC(ksp, &pc));

    PetscCall(PetscTime(&start));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(PetscTime(&finish));
    PetscCall(KSPGetIterationNumber(ksp, &its));

    printf("%d %g\n", its, finish-start);

    free(row);
    free(col);
    free(data);
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(PetscViewerDestroy(&viewer));

    return 0;
}