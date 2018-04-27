#include "CCL_LE_GPU.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>

const int BLOCK = 8;

__device__ int IMin(int a, int b)
{
	return a < b ? a : b;
}

__device__ unsigned char diff(unsigned char a, unsigned char b)
{
	return abs(a - b);
}

__global__ void InitCCL(int labelList[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelList[id] = reference[id] = id;
}

__global__ void Scanning(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, unsigned char threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	unsigned char value = frame[id];
	int label = N;

	if (id - width >= 0 && diff(value, frame[id - width]) <= threshold)
		label = IMin(label, labelList[id - width]);
	if (id + width < N  && diff(value, frame[id + width]) <= threshold)
		label = IMin(label, labelList[id + width]);

	int col = id % width;

	if (col > 0 && diff(value, frame[id - 1]) <= threshold)
		label = IMin(label, labelList[id - 1]);
	if (col + 1 < width  && diff(value, frame[id + 1]) <= threshold)
		label = IMin(label, labelList[id + 1]);

	if (label < labelList[id])
	{
		reference[labelList[id]] = label;
		*markFlag = true;
	}
}

__global__ void scanning8(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, unsigned char threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id = x + y * blockDim.x * gridDim.x;

	if (id >= N)
		return;

	unsigned char value = frame[id];
	int label = N;

	if (id - width >= 0 && diff(value, frame[id - width]) <= threshold)
		label = IMin(label, labelList[id - width]);

	if (id + width < N  && diff(value, frame[id + width]) <= threshold)
		label = IMin(label, labelList[id + width]);

	int col = id % width;
	if (col > 0)
	{
		if (diff(value, frame[id - 1]) <= threshold)
			label = IMin(label, labelList[id - 1]);
		if (id - width - 1 >= 0 && diff(value, frame[id - width - 1]) <= threshold)
			label = IMin(label, labelList[id - width - 1]);
		if (id + width - 1 < N  && diff(value, frame[id + width - 1]) <= threshold)
			label = IMin(label, labelList[id + width - 1]);
	}
	if (col + 1 < width)
	{
		if (diff(value, frame[id + 1]) <= threshold)
			label = IMin(label, labelList[id + 1]);
		if (id - width + 1 >= 0 && diff(value, frame[id - width + 1]) <= threshold)
			label = IMin(label, labelList[id - width + 1]);
		if (id + width + 1 < N  && diff(value, frame[id + width + 1]) <= threshold)
			label = IMin(label, labelList[id + width + 1]);
	}

	if (label < labelList[id])
	{
		reference[labelList[id]] = label;
		*markFlag = true;
	}
}

__global__ void analysis(int labelList[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	int label = labelList[id];
	int ref;
	if (label == id)
	{
		do
		{
			ref = label;
			label = reference[ref];
		} while (ref ^ label);
		reference[id] = label;
	}
}

__global__ void labeling(int labelList[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelList[id] = reference[reference[labelList[id]]];
}



void CCLLEGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold)
{
	auto N = width * height;

	cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&ReferenceOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* markFlagOnDevice;
	cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	dim3 threads(BLOCK, BLOCK);

	InitCCL <<<grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);

	auto initLabel = static_cast<int*>(malloc(sizeof(int) * width * height));

	cudaMemcpy(initLabel, LabelListOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	std::cout << "Init labels:" << std::endl;
	for (auto i = 0; i < height; ++i)
	{
		for (auto j = 0; j < width; ++j)
		{
			std::cout << std::setw(3) << initLabel[i * width + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	free(initLabel);

	while (true)
	{
		auto markFalgOnHost = false;
		cudaMemcpy(markFlagOnDevice, &markFalgOnHost, sizeof(bool), cudaMemcpyHostToDevice);

		if (degreeOfConnectivity == 4)
		{
			Scanning <<< grid, threads >>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, markFlagOnDevice, N, width, height, threshold);
			cudaThreadSynchronize();
		}
		else
			scanning8 <<< grid, threads >>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, markFlagOnDevice, N, width, height, threshold);

		cudaThreadSynchronize();
		cudaMemcpy(&markFalgOnHost, markFlagOnDevice, sizeof(bool), cudaMemcpyDeviceToHost);

		if (markFalgOnHost)
		{
			analysis <<< grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);
			cudaThreadSynchronize();
			labeling <<< grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);
		}
		else
		{
			break;
		}
	}

	cudaMemcpy(labels, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(FrameDataOnDevice);
	cudaFree(LabelListOnDevice);
	cudaFree(ReferenceOnDevice);
}