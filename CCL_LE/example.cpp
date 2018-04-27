#include "CCL_LE_CPU.hpp"
#include "CCL_LE_GPU.cuh"
#include "CCL_NP_GPU.cuh"
#include "CCL_DPL_GPU.cuh"

#include "common.h"

#include <iomanip>
#include <iostream>

using namespace std;

int main()
{
//	const auto width = 9;
//	const auto height = 8;

//	unsigned char data[width * height] =
//	{
//		2,1, 1, 1, 1, 1, 1, 0, 0,
//		2,0, 0, 0, 1, 1, 1, 1, 0,
//		2,0, 0, 0, 1, 1, 1, 1, 0,
//		2,0, 0, 0, 0, 1, 1, 1, 1,
//		2,0, 0, 0, 0, 0, 1, 1, 1,
//		2,0, 0, 0, 1, 1, 1, 1, 1,
//		2,0, 1, 1, 1, 1, 0, 0, 0,
//		2,0, 1, 0, 0, 0, 0, 0, 0
//	};

//	const auto width = 12;
//	const auto height = 8;
//	unsigned char data[width * height] =
//	{
//		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135,
//		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135,
//		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135,
//		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120,
//		135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120,
//		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120,
//		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120,
//		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120
//	};

	const auto width = 32;
	const auto height = 8;
	unsigned char data[width * height] =
	{
		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 135, 120, 120,
		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 135, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120
	};

//	const auto width = 320;
//	const auto height = 256;
//	unsigned char data[width * height] = {0};

//	ifstream fin;
//	fin.open("dataOriginal.txt", ios::in);
//	if (fin.is_open())
//	{
//		int txt;
//		for (auto i = 0; i < width * height; ++i)
//		{
//			fin >> txt;
//			data[i] = static_cast<unsigned char>(txt);
//		}
//		fin.close();
//	}
//	else
//	{
//		cout << "Read Data file fialed" << endl;
//	}

	int labels[width * height] = { 0 };

	cout << "Binary image is : " <<endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << static_cast<int>(data[i * width + j]) << " ";
		}
		cout << endl;
	}
	cout<<endl;

	auto degreeOfConnectivity = 4;
	unsigned char threshold = 0;

	CCLLEGPU ccl;

	CheckPerf(ccl.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold),"CCLLEGPU LE");

	cout << "Label Mesh by CCL LE : " <<endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << labels[i * width + j] << " ";
		}
		cout << endl;
	}

	CCLNPGPU cclnp;
	CheckPerf(cclnp.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold), "CCL_NP_GPU");

	cout << "Label Mesh by CCL NP : " << endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << labels[i * width + j] << " ";
		}
		cout << endl;
	}

	CCLDPLGPU ccldpl;
	CheckPerf(ccldpl.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold), "CCL_DPL_GPU");

	cout << "Label Mesh by CCL DPL : " << endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << labels[i * width + j] << " ";
		}
		cout << endl;
	}

	cout << "Calculate CCL On CPU" <<endl;
	CCLLECPU cclCPU;
	int labelsUseCPU[width * height] = { 0 };
	cclCPU.ccl(data, labelsUseCPU, width, height, degreeOfConnectivity, threshold);

	cout << "Label Mesh  by CPU: " << endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << labelsUseCPU[i * width + j] << " ";
		}
		cout << endl;
	}

//	ofstream fout;
//	fout.open("dataOriginalResult.txt", ios::out);
//	if (fout.is_open())
//	{
//		int txt;
//		for (auto i = 0; i < width * height; ++i)
//		{
//			fout << labels[i] << " ";
//			if ((i + 1) % width == 0)
//				fout << endl;
//		}
//		fin.close();
//	}
//	else
//	{
//		cout << "Write result file fialed" << endl;
//	}

	system("Pause");
    return 0;
}
