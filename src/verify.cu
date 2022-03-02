#include "common/common.cuh"

#include <unistd.h>

using namespace std;

int main(int argc, char *argv[]) {
    // parseargs
    int kernel_id = 0;
    int version_id = 0;

    int opt;
    const char *optstring = "k:v:n::";

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'k':
                sscanf(optarg, "%d", &kernel_id);
                break;
            case 'v':
                sscanf(optarg, "%d", &version_id);
                break;
            default:
                printf("Unknown argument %c\n", opt);
                exit(1);
        }
    }

    printf("\nRunning verification on kernel %d version %d\n", kernel_id, version_id);
    
    // create dataloader 
    cks::common::DataLoader *p_dataloader = cks::common::createDataLoader(kernel_id);

    // for each data, verify it
    for (int i = 0; i < p_dataloader->len(); i++) {
        bool res = cks::common::verifyKernel(kernel_id, version_id, p_dataloader);
        if (!res) {
            printf("Verification on kernel %d version %d failed!\n", kernel_id, version_id);
            cks::common::destroyDataLoader(p_dataloader);
            return 0;
        }
        p_dataloader->step();
        printf(".");
        fflush(stdout);
    }

    // free dataloader
    cks::common::destroyDataLoader(p_dataloader);
    printf("\nVerification: success\n");
}