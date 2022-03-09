#include "common/common.cuh"
#include "data/sgemm.cuh"

#include <random>

#include <unistd.h>

using namespace std;

#define TESTSIZE 512

int main(int argc, char *argv[]) {
    float alpha = 1, beta = 1;
    float A[TESTSIZE][TESTSIZE], B[TESTSIZE][TESTSIZE], C1[TESTSIZE][TESTSIZE], C2[TESTSIZE][TESTSIZE];
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    for (int i = 0; i < TESTSIZE; i++) {
        for (int j = 0; j < TESTSIZE; j++) {
            A[i][j] = distribution(generator);
            B[i][j] = distribution(generator);
            C1[i][j] = 0;
            C2[i][j] = 0;
        }
    }

    cks::sgemm::SgemmArgs args1(TESTSIZE, TESTSIZE, TESTSIZE, &alpha, &beta, (float*)A, (float*)B, (float*)C1);
    cks::sgemm::SgemmArgs args2(TESTSIZE, TESTSIZE, TESTSIZE, &alpha, &beta, (float*)A, (float*)B, (float*)C2);
    cks::sgemm::runKernel(1, &args1);
    cks::sgemm::runKernel(0, &args2);
    
    for (int i = 0; i < TESTSIZE; i++) {
        for (int j = 0; j < TESTSIZE; j++) {
            // printf("%.2f ", C1[i][j]-C2[i][j]);
            if (abs(C1[i][j]-C2[i][j]) > 1e-3) {
                printf("(%d) ", i); break;
            }
        }
        printf("\n");
    }

}