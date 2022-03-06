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

    printf("\nRunning speedtest on kernel %d version %d\n", kernel_id, version_id);
    
    // create dataloader 
    cks::common::DataLoader *p_dataloader = cks::common::createDataLoader(kernel_id);

    // for each data, speedTest it
    for (int i = 0; i < p_dataloader->len(); i++) {

        cks::common::KernelArgs *p_args;
        p_dataloader->loadData(&p_args);

        float res = cks::common::speedTestKernel(kernel_id, version_id, p_args);
        p_dataloader->log(res);

        p_dataloader->freeData(p_args);
        p_dataloader->step();
    }

    // free dataloader
    cks::common::destroyDataLoader(p_dataloader);
}