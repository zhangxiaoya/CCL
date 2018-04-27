#pragma once
#include <host_defines.h>
#include "CCL_DPL_GPU.cuh"
#include <device_launch_parameters.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>

const int BLOCK = 8;

__global__ void init_CCLDPL(int labelOnDevice[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelOnDevice[id] = id;
}

__device__ unsigned char DiffDPL(unsigned char d1, unsigned char d2)
{
	return abs(d1 - d2);
}

__global__ void kernelDPL(int I, unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;
	int H = N / width;
	int S, E, step;
	switch (I)
	{
	case 0:
		if (id >= width)
			return;
		S = id;
		E = width * (H - 1) + id;
		step = width;
		break;
	case 1:
		if (id >= H)
			return;
		S = id * width;
		E = S + width - 1;
		step = 1;
		break;
	case 2:
		if (id >= width) return;
		S = width * (H - 1) + id;
		E = id;
		step = - width;
		break;
	case 3:
		if (id >= H) return;
		S = (id + 1) * width - 1;
		E = id * width;
		step = -1;
		break;
	}

	int label = labelOnDevice[S];
	for (int n = S + step; n != E + step; n += step)
	{
		if (DiffDPL(dataOnDevice[n], dataOnDevice[n - step]) <= threshold && label < labelOnDevice[n])
		{
			labelOnDevice[n] = label;
			*markFlagOnDevice = true;
		}
		else label = labelOnDevice[n];
	}
}

__global__ void kernelDPL8(int I, unsigned char dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;
	int H = N / width;
	int S, E1, E2, step;
	switch (I)
	{
	case 0:
		if (id >= width + H - 1) return;
		if (id < width) S = id;
		else S = (id - width + 1) * width;
		E1 = width - 1; // % W
		E2 = H - 1; // / W
		step = width + 1;
		break;
	case 1:
		if (id >= width + H - 1) return;
		if (id < width) S = width * (H - 1) + id;
		else S = (id - width + 1) * width;
		E1 = width - 1; // % W
		E2 = 0; // / W
		step = -width + 1;
		break;
	case 2:
		if (id >= width + H - 1) return;
		if (id < width) S = width * (H - 1) + id;
		else S = (id - width) * width + width - 1;
		E1 = 0; // % W
		E2 = 0; // / W
		step = -(width + 1);
		break;
	case 3:
		if (id >= width + H - 1) return;
		if (id < width) S = id;
		else S = (id - width + 1) * width + width - 1;
		E1 = 0; // % W
		E2 = H - 1; // / W
		step = width - 1;
		break;
	}

	if (E1 == S % width || E2 == S / width)
		return;
	int label = labelOnDevice[S];
	for (int n = S + step;; n += step)
	{
		if (DiffDPL(dataOnDevice[n], dataOnDevice[n - step]) <= threshold && label < labelOnDevice[n])
		{
			labelOnDevice[n] = label;
			*markFlagOnDevice = true;
		}
		else label = labelOnDevice[n];
		if (E1 == n % width || E2 == n / width)
			break;
	}
}

void CCLDPLGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold)
{
	auto N = width * height;

	cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* markFlagOnDevice;
	cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	dim3 threads(BLOCK, BLOCK);

	init_CCLDPL<<<grid, threads >>>(LabelListOnDevice, width, height);

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

		for (int i = 0; i < 4; i++)
		{
			kernelDPL<<< grid, threads >>>(i, FrameDataOnDevice, LabelListOnDevice, markFlagOnDevice, N, width, height, threshold);
			if (degreeOfConnectivity == 8)
			{
				kernelDPL<<< grid, threads>>>(i, FrameDataOnDevice, LabelListOnDevice, markFlagOnDevice, N, width, height, threshold);
			}
		}
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
