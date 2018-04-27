#pragma once
#include <host_defines.h>

__global__ void init_CCLDPL(int labelOnDevice[], int width, int height);

__device__ unsigned char DiffDPL(unsigned char d1, unsigned char d2);

__global__ void kernelDPL(int I, unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold);

__global__ void kernelDPL8(int I, unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold);

class CCLDPLGPU
{
public:
	explicit CCLDPLGPU(unsigned char* dataOnDevice = nullptr, int* labelListOnDevice = nullptr)
		: FrameDataOnDevice(dataOnDevice),
		  LabelListOnDevice(labelListOnDevice)
	{
	}

	void CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold);

private:
	unsigned char* FrameDataOnDevice;
	int* LabelListOnDevice;
};