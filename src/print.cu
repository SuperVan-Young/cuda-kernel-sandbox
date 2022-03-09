#include "common/common.cuh"
#include "data/sgemm.cuh"

#include <random>

#include <unistd.h>

using namespace std;

#define TESTSIZE 2048

#define A(i,j) A[(i)+(j)*TESTSIZE]
#define B(i,j) B[(i)+(j)*TESTSIZE]
#define C1(i,j) C1[(i)+(j)*TESTSIZE]
#define C2(i,j) C2[(i)+(j)*TESTSIZE]

int main(int argc, char *argv[]) {
    float alpha = 1, beta = 1;
    float *A  = (float *)malloc(sizeof(float) * TESTSIZE * TESTSIZE);
    float *B  = (float *)malloc(sizeof(float) * TESTSIZE * TESTSIZE);
    float *C1 = (float *)malloc(sizeof(float) * TESTSIZE * TESTSIZE);
    float *C2 = (float *)malloc(sizeof(float) * TESTSIZE * TESTSIZE);


    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    for (int i = 0; i < TESTSIZE; i++) {
        for (int j = 0; j < TESTSIZE; j++) {
            A(i,j) = distribution(generator);
            B(i,j) = distribution(generator);
            C1(i,j) = 0;
            C2(i,j) = 0;
        }
    }

    cks::sgemm::SgemmArgs args1(TESTSIZE, TESTSIZE, TESTSIZE, &alpha, &beta, (float*)A, (float*)B, (float*)C1);
    cks::sgemm::SgemmArgs args2(TESTSIZE, TESTSIZE, TESTSIZE, &alpha, &beta, (float*)A, (float*)B, (float*)C2);
    cks::sgemm::runKernel(5, &args1);
    cks::sgemm::runKernel(1, &args2);
    
    for (int i = 0; i < TESTSIZE; i++) {
        for (int j = 0; j < TESTSIZE; j++) {
            // printf("%.2f ", C1[i][j]-C2[i][j]);
            if (abs(C1(i,j)-C2(i,j)) > 1e-3) {
                printf("(%d) \n", i); break;
            }
        }
        // printf("\n");
    }

    // for (int i = 0; i < TESTSIZE; i++) {
    //     for (int j = 0; j < TESTSIZE; j++) {
    //         printf("%.0f ", C1(i,j)-C2(i,j));
    //     }
    //     printf("\n");
    // }

    free(A);
    free(B);
    free(C1);
    free(C2);

}