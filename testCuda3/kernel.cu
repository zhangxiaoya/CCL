#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <ostream>
#include <iostream>

const int BLOCK = 8;

using namespace std;

inline double get_time()
{
	return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

__device__ int IMin(int a, int b)
{
	return a < b ? a : b;
}

__global__ void init_CCL(int L[], int R[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y >= height)
		return;

	int id = x + y * width;

	L[id] = R[id] = id;
}

__device__ int diff(int d1, int d2)
{
	return abs(((d1 >> 16) & 0xff) - ((d2 >> 16) & 0xff)) + abs(((d1 >> 8) & 0xff) - ((d2 >> 8) & 0xff)) + abs((d1 & 0xff) - (d2 & 0xff));
}

__global__ void scanning(int D[], int L[], int R[], bool* m, int N, int width, int height, int th)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id = x + y * blockDim.x * gridDim.x;
	if (id >= N)
		return;

	int Did = D[id];
	int label = N;
	if (id - width >= 0 && diff(Did, D[id - width]) <= th)
		label = IMin(label, L[id - width]);
	if (id + width < N  && diff(Did, D[id + width]) <= th)
		label = IMin(label, L[id + width]);
	int r = id % width;
	if (r           && diff(Did, D[id - 1]) <= th)
		label = IMin(label, L[id - 1]);
	if (r + 1 != width  && diff(Did, D[id + 1]) <= th)
		label = IMin(label, L[id + 1]);

	if (label < L[id])
	{
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void scanning8(int D[], int L[], int R[], bool* m, int N, int width, int height, int th)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id = x + y * blockDim.x * gridDim.x;

	if (id >= N) return;

	int Did = D[id];
	int label = N;
	if (id - width >= 0 && diff(Did, D[id - width]) <= th) label = IMin(label, L[id - width]);
	if (id + width < N  && diff(Did, D[id + width]) <= th) label = IMin(label, L[id + width]);
	int r = id % width;
	if (r)
	{
		if (diff(Did, D[id - 1]) <= th) label = IMin(label, L[id - 1]);
		if (id - width - 1 >= 0 && diff(Did, D[id - width - 1]) <= th) label = IMin(label, L[id - width - 1]);
		if (id + width - 1 < N  && diff(Did, D[id + width - 1]) <= th) label = IMin(label, L[id + width - 1]);
	}
	if (r + 1 != width) {
		if (diff(Did, D[id + 1]) <= th) label = IMin(label, L[id + 1]);
		if (id - width + 1 >= 0 && diff(Did, D[id - width + 1]) <= th) label = IMin(label, L[id - width + 1]);
		if (id + width + 1 < N  && diff(Did, D[id + width + 1]) <= th) label = IMin(label, L[id + width + 1]);
	}

	if (label < L[id])
	{
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void analysis(int D[], int L[], int R[], int width, int height, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	int label = L[id];
	int ref;
	if (label == id)
	{
		do
		{
			ref = label;
			label = R[ref];
		}
		while (ref ^ label);
		R[id] = label;
	}
}

__global__ void labeling(int D[], int L[], int R[], int width, int height, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	L[id] = R[R[L[id]]];
}

class CCL
{
public:
	std::vector<int> cuda_ccl(std::vector<int>& image, int W, int height, int degree_of_connectivity, int threshold);

private:
	int* Dd;
	int* Ld;
	int* Rd;
};

vector<int> CCL::cuda_ccl(std::vector<int>& image, int width, int height, int degree_of_connectivity, int threshold)
{

	vector<int> result;
	int* D = static_cast<int*>(&image[0]);
	int N = image.size();

	cudaMalloc((void**)&Ld, sizeof(int) * N);
	cudaMalloc((void**)&Rd, sizeof(int) * N);
	cudaMalloc((void**)&Dd, sizeof(int) * N);

	cudaMemcpy(Dd, D, sizeof(int) * N, cudaMemcpyHostToDevice);

	bool* md;
	cudaMalloc((void**)&md, sizeof(bool));

	int gridWidth = static_cast<int>(sqrt(static_cast<double>(N) / BLOCK)) + 1;

	dim3 grid((width + BLOCK - 1)/ BLOCK, (height + BLOCK -1)/BLOCK);
	dim3 threads(BLOCK,BLOCK);

	init_CCL<<<grid, threads>>>(Ld, Rd,width,height);

	int* t = (int*)malloc(sizeof(int) * width * height);

	cudaMemcpy(t, Ld, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
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

		if (degree_of_connectivity == 4)
			scanning<<<grid, threads>>>(Dd, Ld, Rd, md, N, width, height, threshold);
		else
			scanning8<<<grid, threads >>>(Dd, Ld, Rd, md, N, width, height, threshold);

		cudaMemcpy(&m, md, sizeof(bool), cudaMemcpyDeviceToHost);

		if (m)
		{
			analysis<<<grid, threads>>>(Dd, Ld, Rd, width, height, N);
			//cudaThreadSynchronize();
			labeling<<<grid, threads>>>(Dd, Ld, Rd, width, height, N);
		}
		else
		{
			break;
		}
	}


	cudaMemcpy(D, Ld, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(Dd);
	cudaFree(Ld);
	cudaFree(Rd);

	result.swap(image);
	return result;
}

int main()
{
	const int width = 10;
	const int height = 8;

	int data[width * height] =
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

	vector<int> image(data, data + width * height);

	cout << "binary image" <<endl;
	for (auto i = 0; i < static_cast<int>(image.size()) / width; i++)
	{
		for (auto j = 0; j < width; j++)
			cout << image[i * width + j] << " ";

		cout << endl;
	}
	cout<<endl;

	auto degree_of_connectivity = 4;
	auto threshold = 0;

	CCL ccl;

	auto start = get_time();
	auto result = ccl.cuda_ccl(image, width, height, degree_of_connectivity, threshold);
	auto end = get_time();

	cerr << "Time: " << end - start << endl;

	cout << result.size() << endl;
	cout << width << endl;

	for (auto i = 0; i < static_cast<int>(result.size()) / width; i++)
	{
		for (auto j = 0; j < width; j++)
			cout << result[i * width + j] << " ";

		cout << endl;
	}

	system("Pause");
    return 0;
}
