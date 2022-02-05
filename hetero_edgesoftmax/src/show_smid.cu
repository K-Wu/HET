//From https://stackoverflow.com/questions/70547875/strange-behaviors-of-cuda-kernel-with-infinite-loop-on-different-nvidia-gpu
#include <cstdio>
__global__ void loop(void) {
    int smid = -1;
    int nsmid = -1;
    if (threadIdx.x == 0) {
        asm volatile("mov.u32 %0, %%smid;": "=r"(smid));
        asm volatile("mov.u32 %0, %%nsmid;": "=r"(nsmid));
        printf("smid: %d, smnid: %d, blockidx: %d\n", smid,nsmid, blockIdx.x);
    }
}

int main() {
    loop<<<256, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
