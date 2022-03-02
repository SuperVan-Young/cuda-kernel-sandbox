# CUDA Kernel Sandbox (CKS)

A general-purpose, light-weighted platform to run, validate and speed-test 
GPU kernel functions implemented with multiple versions.

---

## How to run the code
```
mkdir -p build
cd ./build
cmake ..
make install
cd ..
bash run.sh
```

## Experiment Settings
- CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- GPU: Nvidia Tesla V100

## References
[Optimizing-SGEMM-on-NVIDIA-Turing-GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)