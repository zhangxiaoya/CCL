#pragma once
#include "CCL_NP_GPU.cuh"

#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

const int BLOCK = 8;

__device__ int IMinNP(int a, int b)
{
	return a < b ? a : b;
}

__device__ unsigned char DiffNP(unsigned char d1, unsigned char d2)
{
	return abs(d1 - d2);
}

__global__ void InitCCL(int labelList[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelList[id] = id;
}

__global__ void kernel(unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N,int width, int height, int threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	int Did = dataOnDevice[id];
	int label = N;
	// up and down
	if (id - width >= 0 && DiffNP(Did, dataOnDevice[id - width]) <= threshold)
		label = IMinNP(label, labelOnDevice[id - width]);
	if (id + width < N && DiffNP(Did, dataOnDevice[id + width]) <= threshold)
		label = IMinNP(label, labelOnDevice[id + width]);
	// left and right
	int r = id % width;
	if (r && DiffNP(Did, dataOnDevice[id - 1]) <= threshold)
		label = IMinNP(label, labelOnDevice[id - 1]);
	if (r + 1 != width && DiffNP(Did, dataOnDevice[id + 1]) <= threshold)
		label = IMinNP(label, labelOnDevice[id + 1]);

	if (label < labelOnDevice[id])
	{
		//atomicMin(&R[labelOnDevice[id]], label);
		labelOnDevice[id] = label;
		*markFlagOnDevice = true;
	}
}

__global__ void kernel8(unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	int Did = dataOnDevice[id];
	int label = N;
	if (id - width >= 0 && DiffNP(Did, dataOnDevice[id - width]) <= threshold)
		label = IMinNP(label, labelOnDevice[id - width]);
	if (id + width < N && DiffNP(Did, dataOnDevice[id + width]) <= threshold)
		label = IMinNP(label, labelOnDevice[id + width]);
	int r = id % width;
	if (r)
	{
		if (DiffNP(Did, dataOnDevice[id - 1]) <= threshold) label = IMinNP(label, labelOnDevice[id - 1]);
		if (id - width - 1 >= 0 && DiffNP(Did, dataOnDevice[id - width - 1]) <= threshold) label = IMinNP(label, labelOnDevice[id - width - 1]);
		if (id + width - 1 < N && DiffNP(Did, dataOnDevice[id + width - 1]) <= threshold) label = IMinNP(label, labelOnDevice[id + width - 1]);
	}
	if (r + 1 != width)
	{
		if (DiffNP(Did, dataOnDevice[id + 1]) <= threshold) label = IMinNP(label, labelOnDevice[id + 1]);
		if (id - width + 1 >= 0 && DiffNP(Did, dataOnDevice[id - width + 1]) <= threshold) label = IMinNP(label, labelOnDevice[id - width + 1]);
		if (id + width + 1 < N && DiffNP(Did, dataOnDevice[id + width + 1]) <= threshold) label = IMinNP(label, labelOnDevice[id + width + 1]);
	}

	if (label < labelOnDevice[id])
	{
		//atomicMin(&R[labelOnDevice[id]], label);
		labelOnDevice[id] = label;
		*markFlagOnDevice = true;
	}
}

void CCLNPGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold)
{
	auto N = width * height;

	cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* markFlagOnDevice;
	cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	dim3 threads(BLOCK, BLOCK);

	InitCCL <<<grid, threads >>>(LabelListOnDevice, width, height);

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
			kernel<<< grid, threads >>>(FrameDataOnDevice, LabelListOnDevice, markFlagOnDevice, N, width, height, threshold);
			cudaThreadSynchronize();
		}
		else
			kernel8 <<< grid, threads >>>(FrameDataOnDevice, LabelListOnDevice, markFlagOnDevice, N, width, height, threshold);

		cudaThreadSynchronize();
		cudaMemcpy(&markFalgOnHost, markFlagOnDevice, sizeof(bool), cudaMemcpyDeviceToHost);

		if (markFalgOnHost)
		{
			cudaThreadSynchronize();
		}
		else
		{
			break;
		}
	}

	cudaMemcpy(labels, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(FrameDataOnDevice);
	cudaFree(LabelListOnDevice);
}
