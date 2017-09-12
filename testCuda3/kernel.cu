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

__global__ void init_CCL(int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[id] = id;
}

__device__ int diff(int d1, int d2)
{
	return abs(((d1 >> 16) & 0xff) - ((d2 >> 16) & 0xff)) + abs(((d1 >> 8) & 0xff) - ((d2 >> 8) & 0xff)) + abs((d1 & 0xff) - (d2 & 0xff));
}

__global__ void scanning(int D[], int L[], int R[], bool* m, int N, int W, int th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int Did = D[id];
	int label = N;
	if (id - W >= 0 && diff(Did, D[id - W]) <= th) label = IMin(label, L[id - W]);
	if (id + W < N  && diff(Did, D[id + W]) <= th) label = IMin(label, L[id + W]);
	int r = id % W;
	if (r           && diff(Did, D[id - 1]) <= th) label = IMin(label, L[id - 1]);
	if (r + 1 != W  && diff(Did, D[id + 1]) <= th) label = IMin(label, L[id + 1]);

	if (label < L[id]) {
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void scanning8(int D[], int L[], int R[], bool* m, int N, int W, int th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int Did = D[id];
	int label = N;
	if (id - W >= 0 && diff(Did, D[id - W]) <= th) label = IMin(label, L[id - W]);
	if (id + W < N  && diff(Did, D[id + W]) <= th) label = IMin(label, L[id + W]);
	int r = id % W;
	if (r) {
		if (diff(Did, D[id - 1]) <= th) label = IMin(label, L[id - 1]);
		if (id - W - 1 >= 0 && diff(Did, D[id - W - 1]) <= th) label = IMin(label, L[id - W - 1]);
		if (id + W - 1 < N  && diff(Did, D[id + W - 1]) <= th) label = IMin(label, L[id + W - 1]);
	}
	if (r + 1 != W) {
		if (diff(Did, D[id + 1]) <= th) label = IMin(label, L[id + 1]);
		if (id - W + 1 >= 0 && diff(Did, D[id - W + 1]) <= th) label = IMin(label, L[id - W + 1]);
		if (id + W + 1 < N  && diff(Did, D[id + W + 1]) <= th) label = IMin(label, L[id + W + 1]);
	}

	if (label < L[id]) {
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void analysis(int D[], int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int label = L[id];
	int ref;
	if (label == id) {
		do { label = R[ref = label]; } while (ref ^ label);
		R[id] = label;
	}
}

__global__ void labeling(int D[], int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[R[L[id]]];
}

class CCL
{
public:
	std::vector<int> cuda_ccl(std::vector<int>& image, int W, int degree_of_connectivity, int threshold);

private:
	int* Dd;
	int* Ld;
	int* Rd;
};

vector<int> CCL::cuda_ccl(std::vector<int>& image, int W, int degree_of_connectivity, int threshold)
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

	int width = static_cast<int>(sqrt(static_cast<double>(N) / BLOCK)) + 1;
	dim3 grid(width, width, 1);
	dim3 threads(BLOCK, 1, 1);

	init_CCL<<<grid, threads>>>(Ld, Rd, N);

	for (;;)
	{
		bool m = false;
		cudaMemcpy(md, &m, sizeof(bool), cudaMemcpyHostToDevice);

		if (degree_of_connectivity == 4)
			scanning<<<grid, threads>>>(Dd, Ld, Rd, md, N, W, threshold);
		else
			scanning8<<<grid, threads >>>(Dd, Ld, Rd, md, N, W, threshold);

		cudaMemcpy(&m, md, sizeof(bool), cudaMemcpyDeviceToHost);

		if (m)
		{
			analysis<<<grid, threads>>>(Dd, Ld, Rd, N);
			//cudaThreadSynchronize();
			labeling<<<grid, threads>>>(Dd, Ld, Rd, N);
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
	const int width = 8;
	const int height = 8;

	int data[width * height] =
	{
		1, 1, 1, 1, 1, 1, 0, 0,
		0, 0, 0, 1, 1, 1, 1, 0,
		0, 0, 0, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 1, 1, 1, 1, 1,
		0, 1, 1, 1, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0
	};

	std::vector<int> image(data, data + width * height);

	auto W = 8, degree_of_connectivity = 4, threshold = 0;

	CCL ccl;

	auto start = get_time();
	auto result(ccl.cuda_ccl(image, W, degree_of_connectivity, threshold));
	auto end = get_time();
	std::cerr << "Time: " << end - start << std::endl;

	std::cout << result.size() << std::endl; /// number of pixels
	std::cout << W << std::endl; /// width

	for (auto i = 0; i < static_cast<int>(result.size()) / W; i++)
	{
		for (auto j = 0; j < W; j++)
			std::cout << result[i*W + j] << " ";
		std::cout << std::endl;
	}
	system("Pause");
    return 0;
}
