#include <bits/stdc++.h>
#include <cuda_runtime.h>



#define CHECK(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

int main ( ) {
	int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "ʹ��GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM��������" << devProp.multiProcessorCount << std::endl;
    std::cout << "ÿ���߳̿�Ĺ����ڴ��С��" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "ÿ���߳̿������߳�����" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "ÿ��EM������߳�����" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "ÿ��SM������߳�������" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

	return 0 ;
}
