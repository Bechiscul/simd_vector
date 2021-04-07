#include <benchmark/benchmark.h>
#include <vector/vector.hpp>

#include <cstdlib>
#include <ctime>

#define kVectorSize 1000

Vector<double, kVectorSize> rand_vec()
{
	Vector<double, kVectorSize> vec;

	srand(time(0));

	for (size_t i = 0; i < kVectorSize; ++i)
	{
		vec[i] = (double)(rand()) * (double)10;
	}

	return vec;
}

void SIMD(benchmark::State& state)
{
	Vector<double, kVectorSize> vec1 = rand_vec();
	Vector<double, kVectorSize> vec2 = rand_vec();

	for (auto _ : state)
	{
		benchmark::DoNotOptimize(vec1.dot(vec2));
	}
}

BENCHMARK(SIMD);

void Normal(benchmark::State& state)
{
	Vector<double, kVectorSize> vec1 = rand_vec();
	Vector<double, kVectorSize> vec2 = rand_vec();

	for (auto _ : state)
	{
		benchmark::DoNotOptimize(vec1.normal_dot(vec2));
	}
}

BENCHMARK(Normal);

BENCHMARK_MAIN();