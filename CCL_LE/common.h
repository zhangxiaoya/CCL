#pragma once
#define CheckPerf(call, message)                                                                             \
{                                                                                                            \
	LARGE_INTEGER t1, t2, tc;                                                                                \
	QueryPerformanceFrequency(&tc);                                                                          \
	QueryPerformanceCounter(&t1);                                                                            \
	call;                                                                                                    \
	QueryPerformanceCounter(&t2);                                                                            \
	printf("Operation of %20s Use Time:%f\n", message, (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);       \
};
