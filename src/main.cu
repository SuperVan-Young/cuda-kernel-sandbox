#include "common/common.cuh"

#include <unistd.h>

using namespace std;

int main(int argc, char *argv[]) {
    // parseargs
    int kernel_id = 0;
    int version_id = 0;

    int opt;
    const char *optstring = "k:v:";

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
    
    // create dataloader 
    cks::common::DataLoader *p_dataloader = cks::common::createDataLoader(kernel_id);

    // for each data, verify and speedTest it
    for (int i = 0; i < p_dataloader->len(); p_dataloader->step()) {
        cks::common::KernelArgs *p_args;
        p_dataloader->loadData(&p_args);
        cks::common::runKernel(kernel_id, version_id, p_args);
        p_dataloader->freeData(p_args);
    }

    // free dataloader
    delete p_dataloader;
}