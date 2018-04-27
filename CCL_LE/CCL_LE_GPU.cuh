#include "cuda_runtime.h"

__device__ int IMin(int a, int b);

__device__ unsigned char diff(unsigned char a, unsigned char b);

__global__ void InitCCL(int labelList[], int reference[], int width, int height);

__global__ void Scanning(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, unsigned char threshold);

__global__ void scanning8(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, unsigned char threshold);

__global__ void analysis(int labelList[], int reference[], int width, int height);

__global__ void labeling(int labelList[], int reference[], int width, int height);

class CCLLEGPU
{
public:
	explicit CCLLEGPU(unsigned char* dataOnDevice = nullptr, int* labelListOnDevice = nullptr, int* referenceOnDevice = nullptr)
		: FrameDataOnDevice(dataOnDevice),
		  LabelListOnDevice(labelListOnDevice),
		  ReferenceOnDevice(referenceOnDevice)
	{
	}

	void CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold);

private:
	unsigned char* FrameDataOnDevice;
	int* LabelListOnDevice;
	int* ReferenceOnDevice;
};
