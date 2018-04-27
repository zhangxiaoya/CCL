#pragma once
#include <functional>

class Helper
{
public:
	static void InitCCL(int labelList[], int reference[], int N);

	static unsigned char Diff(unsigned char a, unsigned char b);

	static int IMin(int a, int b);
};

inline void Helper::InitCCL(int labelList[], int reference[], int N)
{
	for (auto id = 0; id < N; id++)
	{
		labelList[id] = reference[id] = id;
	}
}

inline unsigned char Helper::Diff(unsigned char a, unsigned char b)
{
	return abs(a - b);
}

inline int Helper::IMin(int a, int b)
{
	return a < b ? a : b;
}

class CCLLECPU
{
public:
	void ccl(unsigned char* image, int* labelList, int width, int height, int degree_of_connectivity, unsigned char threshold) const;

private:
	inline bool scanning(unsigned char* frameData, int* labelList, int* reference, bool& modificationFlag, int N, int widht, unsigned char threshold) const;

	inline bool scanning8(unsigned char* frameData, int* labelList, int* reference, bool& modificationFlag, int N, int width, unsigned char threshold) const;

	inline void analysis(int* labelList, int* reference, int N) const;

	inline void labeling(int* labelList, int* reference, int N) const;
};


inline void CCLLECPU::ccl(unsigned char* image, int* labelList, int width, int height, int degreeOfConnectivity, unsigned char threshold) const
{
	if(image == nullptr || labelList == nullptr)
		return;

	auto N = width * height;
	auto labelListInside = new int[N];
	auto reference = new int[N];

	Helper::InitCCL(labelListInside, reference, N);

	while (true)
	{
		auto modificationFlag = false;
		if (degreeOfConnectivity == 4)
		{
			scanning(image, labelListInside, reference, modificationFlag, N, width, threshold);
		}
		else
		{
			scanning8(image, labelListInside, reference, modificationFlag, N, width, threshold);
		}
		if (modificationFlag)
		{
			analysis(labelListInside, reference, N);
			labeling(labelListInside, reference, N);
		}
		else
			break;
	}

	memcpy(labelList, labelListInside, sizeof(int) * width * height);

	delete[] labelListInside;
	delete[] reference;
}

inline bool CCLLECPU::scanning(unsigned char* frameData, int* labelList, int* reference, bool& modificationFlag, int N, int width, unsigned char threshold) const
{
	for (auto id = 0; id < N; id++)
	{
		auto value = frameData[id];
		auto label = N;

		if (id - width >= 0 && Helper::Diff(value, frameData[id - width]) <= threshold)
			label = Helper::IMin(label, labelList[id - width]);
		if (id + width < N && Helper::Diff(value, frameData[id + width]) <= threshold)
			label = Helper::IMin(label, labelList[id + width]);

		auto col = id % width;

		if (col > 0 && Helper::Diff(value, frameData[id - 1]) <= threshold)
			label = Helper::IMin(label, labelList[id - 1]);
		if (col + 1 != width && Helper::Diff(value, frameData[id + 1]) <= threshold)
			label = Helper::IMin(label, labelList[id + 1]);

		if (label < labelList[id])
		{
			reference[labelList[id]] = label;
			modificationFlag = true;
		}
	}

	return modificationFlag;
}

inline bool CCLLECPU::scanning8(unsigned char* frameData, int* labelList, int* reference, bool& modificationFlag, int N, int width, unsigned char threshold) const
{
	for (auto id = 0; id < N; id++)
	{
		auto value = frameData[id];
		auto label = N;

		if (id - width >= 0 && Helper::Diff(value, frameData[id - width]) <= threshold)
			label = Helper::IMin(label, labelList[id - width]);
		if (id + width < N && Helper::Diff(value, frameData[id + width]) <= threshold)
			label = Helper::IMin(label, labelList[id + width]);

		auto col = id % width;
		if (col > 0)
		{
			if (Helper::Diff(value, frameData[id - 1]) <= threshold)
				label = Helper::IMin(label, labelList[id - 1]);
			if (id - width - 1 >= 0 && Helper::Diff(value, frameData[id - width - 1]) <= threshold)
				label = Helper::IMin(label, labelList[id - width - 1]);
			if (id + width - 1 < N && Helper::Diff(value, frameData[id + width - 1]) <= threshold)
				label = Helper::IMin(label, labelList[id + width - 1]);
		}
		if (col + 1 != width)
		{
			if (Helper::Diff(value, frameData[id + 1]) <= threshold)
				label = Helper::IMin(label, labelList[id + 1]);
			if (id - width + 1 >= 0 && Helper::Diff(value, frameData[id - width + 1]) <= threshold)
				label = Helper::IMin(label, labelList[id - width + 1]);
			if (id + width + 1 < N && Helper::Diff(value, frameData[id + width + 1]) <= threshold)
				label = Helper::IMin(label, labelList[id + width + 1]);
		}

		if (label < labelList[id])
		{
			reference[labelList[id]] = label;
			modificationFlag = true;
		}
	}

	return modificationFlag;
}

inline void CCLLECPU::analysis(int* labelList, int* reference, int N) const
{
	for (auto id = 0; id < N; id++)
	{
		auto label = labelList[id];
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
}

inline void CCLLECPU::labeling(int* labelList, int* reference, int N) const
{
	for (auto id = 0; id < N; id++)
	{
		labelList[id] = reference[reference[labelList[id]]];
	}
}
