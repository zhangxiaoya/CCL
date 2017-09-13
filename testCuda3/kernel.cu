#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <ostream>
#include <iostream>
#include <iomanip>

const int BLOCK = 8;

using namespace std;

inline double get_time()
{
	return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

__device__ unsigned char IMin(unsigned char a, unsigned char b)
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

	if(x >= width || y >= height)
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

	if (col > 0           && diff(value, frame[id - 1]) <= threshold)
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

	if (id >= N) return;

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
		}
		while (ref ^ label);
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

class CCL
{
public:
	explicit CCL(unsigned char* dataOnDevice = nullptr, int* labelListOnDevice=nullptr, int* referenceOnDevice= nullptr)
		: FrameDataOnDevice(dataOnDevice),
		  LabelListOnDevice(labelListOnDevice),
		  ReferenceOnDevice(referenceOnDevice)
	{
	}

	vector<int> cuda_ccl(vector<unsigned char>& image, int width, int height, int degreeOfConnectivity, unsigned char threshold);

private:
	unsigned char* FrameDataOnDevice;
	int* LabelListOnDevice;
	int* ReferenceOnDevice;
};

vector<int> CCL::cuda_ccl(vector<unsigned char>& image, int width, int height, int degreeOfConnectivity, unsigned char threshold)
{

	vector<int> result;
	vector<int> tempResult;
	tempResult.resize(image.size());
	int* temp = static_cast<int*>(&tempResult[0]);
	unsigned char* D = static_cast<unsigned char*>(&image[0]);
	int N = image.size();

	cudaMalloc((void**)&LabelListOnDevice, sizeof(int) * N);
	cudaMalloc((void**)&ReferenceOnDevice, sizeof(int) * N);
	cudaMalloc((void**)&FrameDataOnDevice, sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, D, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* md;
	cudaMalloc((void**)&md, sizeof(bool));

	int gridWidth = static_cast<int>(sqrt(static_cast<double>(N) / BLOCK)) + 1;

	dim3 grid((width + BLOCK - 1)/ BLOCK, (height + BLOCK -1)/BLOCK);
	dim3 threads(BLOCK,BLOCK);

	InitCCL<<<grid, threads>>>(LabelListOnDevice, ReferenceOnDevice,width,height);

	int* t = (int*)malloc(sizeof(int) * width * height);

	cudaMemcpy(t, LabelListOnDevice, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	for(auto i = 0;i<width * height;++i)
	{
		cout << t[i] << " ";
		if ((i+1) % width == 0)
			cout << endl;
	}

	while (true)
	{
		bool m = false;
		cudaMemcpy(md, &m, sizeof(bool), cudaMemcpyHostToDevice);

		if (degreeOfConnectivity == 4)
			Scanning<<<grid, threads>>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, md, N, width, height, threshold);
		else
			scanning8<<<grid, threads >>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, md, N, width, height, threshold);

		cudaMemcpy(&m, md, sizeof(bool), cudaMemcpyDeviceToHost);

		if (m)
		{
			analysis<<<grid, threads>>>(LabelListOnDevice, ReferenceOnDevice, width, height);
			//cudaThreadSynchronize();
			labeling<<<grid, threads>>>(LabelListOnDevice, ReferenceOnDevice, width, height);
		}
		else
		{
			break;
		}
	}


	cudaMemcpy(temp, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(FrameDataOnDevice);
	cudaFree(LabelListOnDevice);
	cudaFree(ReferenceOnDevice);

	result.swap(tempResult);
	return result;
}

int main()
{
	const auto width = 10;
	const auto height = 8;

	unsigned char data[width * height] =
	{
		1,1,1, 1, 1, 1, 1, 1, 0, 0,
		0,0,0, 0, 0, 1, 1, 1, 1, 0,
		0,0,0, 0, 0, 1, 1, 1, 1, 0,
		0,0,0, 0, 0, 0, 1, 1, 1, 1,
		0,0,0, 0, 0, 0, 0, 1, 1, 1,
		0,0,0, 0, 0, 1, 1, 1, 1, 1,
		0,0,0, 1, 1, 1, 1, 0, 0, 0,
		0,0,0, 1, 0, 0, 0, 0, 0, 0
	};

	vector<unsigned char> image(data, data + width * height);

	cout << "binary image" <<endl;
	for (auto i = 0; i < image.size() / width; i++)
	{
		for (auto j = 0; j < width; j++)
			cout << static_cast<int>(image[i * width + j]) << " ";

		cout << endl;
	}
	cout<<endl;

	auto degreeOfConnectivity = 4;
	unsigned char threshold = 0;

	CCL ccl;

	auto start = get_time();
	auto result = ccl.cuda_ccl(image, width, height, degreeOfConnectivity, threshold);
	auto end = get_time();

	cerr << "Time: " << end - start << endl;

	cout << result.size() << endl;
	cout << width << endl;

	for (auto i = 0; i < result.size() / width; i++)
	{
		for (auto j = 0; j < width; j++)
			cout << setw(3)<< result[i * width + j] << " ";

		cout << endl;
	}

	system("Pause");
    return 0;
}
