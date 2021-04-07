#include <cassert>
#include <cstddef>
#include <cstdint>

#include <concepts>
#include <initializer_list>

// AVX intrinsics.
#include <immintrin.h>

template <typename T> concept Integral = requires() { std::is_integral_v<T>; };

template <Integral T, size_t Size> class Vector {
public:
	using value_type = T;
	using size_type = size_t;
	using reference = T&;
	using const_reference = const T&;

public:
	inline constexpr Vector() noexcept {}
	inline constexpr Vector(std::initializer_list<value_type> ilist) noexcept {
		assert(ilist.size() <= Size);

		for (size_type i = 0; i < ilist.size(); ++i) {
			data_[i] = *(ilist.begin() + i);
		}
	}

	inline constexpr value_type dot(const Vector& other) noexcept {
		value_type dot = 0;
		for (size_type i = 0; i < Size; ++i)
			dot += data_[i] * other.data_[i];

		return dot;
	}

private:
	value_type data_[Size] = {};
};

template <size_t Size> class Vector<double, Size> {
public:
	using value_type = double;
	using size_type = size_t;
	using reference = double&;
	using const_reference = const double&;

public:
	inline constexpr Vector() noexcept {}
	inline constexpr Vector(std::initializer_list<value_type> ilist) noexcept {
		assert(ilist.size() <= Size);

		for (size_type i = 0; i < ilist.size(); ++i) {
			data_[i] = *(ilist.begin() + i);
		}
	}

	inline constexpr reference operator[](size_type i) noexcept
	{
		return data_[i];
	}

	inline constexpr value_type dot(const Vector& other) noexcept {
		__m256d vec1 = _mm256_setzero_pd();
		for (size_type i = 0; i < Size / 4; ++i) {
			__m256d vec2 = _mm256_loadu_pd(data_ + i * 4);
			__m256d vec3 = _mm256_loadu_pd(other.data_ + i * 4);
			__m256d vec4 = _mm256_mul_pd(vec2, vec3);

			vec1 = _mm256_add_pd(vec1, vec4);
		}

		value_type rest = 0;
		for (size_type i = Size - Size % 4; i < Size; ++i) {
			rest += data_[i] * other.data_[i];
		}

		__m256d temp = _mm256_hadd_pd(vec1, vec1);
		__m128d sum_high = _mm256_extractf128_pd(temp, 1);
		__m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(temp));

		return rest + ((double*)&result)[0];
	}

	inline constexpr value_type normal_dot(const Vector& other) noexcept {
		value_type dot = 0;
		for (size_type i = 0; i < Size; ++i)
			dot += data_[i] * other.data_[i];

		return dot;
	}

private:
	value_type data_[Size] = {};
};