#pragma once
#include <host_defines.h>

__device__ int IMinNP(int a, int b);

__device__ unsigned char DiffNP(unsigned char d1, unsigned char d2);

__global__ void InitCCL(int labelList[], int width, int height);

__global__ void kernel(unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold);

__global__ void kernel8(unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold);

class CCLNPGPU
{
public:
	explicit CCLNPGPU(unsigned char* dataOnDevice = nullptr, int* labelListOnDevice = nullptr)
		: FrameDataOnDevice(dataOnDevice),
		  LabelListOnDevice(labelListOnDevice)
	{
	}

	void CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold);

private:
	unsigned char* FrameDataOnDevice;
	int* LabelListOnDevice;
};