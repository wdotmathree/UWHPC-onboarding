#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <sys/mman.h>

#include <cstddef>
#include <thread>
#include <xmmintrin.h>

class Grid;

class Proxy {
private:
	Grid *g;
	size_t idx;

public:
	Proxy(Grid *g, size_t idx) : g(g), idx(idx) {}

	operator double() const;
	Proxy operator=(double);
};

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
	size_t rows_;
	size_t cols_;
	size_t size_;

	double *data_;
	uint32_t *fixed_;

	size_t max_steps_ = 0;
	size_t num_steps_ = 0;
	double min_ = +INFINITY;
	double max_ = -INFINITY;
	double quant_;
	double dequant_;
	bool bad_ = false;

	friend class Proxy;

public:
	Grid(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
		constexpr size_t align = 0x200000; // THP page size
		size_ = rows * cols * sizeof(double);
		size_t fixed_size = rows * cols * sizeof(uint32_t);
		size_t align_fixed = (32 - size_ % 32) % 32;
		size_ += align_fixed + fixed_size;
		size_t req = size_ + align;

		void *raw = mmap(nullptr, req, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
		if (raw == MAP_FAILED)
			throw std::bad_alloc();

		uintptr_t addr = (uintptr_t)raw;
		uintptr_t aligned = (addr + align - 1) & ~(align - 1);
		size_t front_slack = aligned - addr;
		size_t back_slack = req - front_slack - size_;

		if (front_slack)
			munmap(raw, front_slack);
		if (back_slack)
			munmap((void *)(aligned + size_), back_slack);

		raw = (void *)aligned;
		data_ = (double *)aligned;
		fixed_ = (uint32_t *)(aligned + rows * cols * sizeof(double) + align_fixed);
		madvise(data_, size_, MADV_SEQUENTIAL | MADV_UNMERGEABLE | MADV_HUGEPAGE | MADV_COLLAPSE);
		memset(data_, 0, size_);
	}

	~Grid() {
		munmap(data_, size_);
	}

	Grid(const Grid &) = delete;
	Grid &operator=(const Grid &) = delete;

	Proxy operator()(size_t i, size_t j) {
		return {this, i * cols_ + j};
	}
	double operator()(size_t i, size_t j) const {
		size_t idx = i * cols_ + j;
		if (bad_ || num_steps_ >= max_steps_)
			return data_[idx];

		return fixed_[idx] * dequant_ + min_;
	}

	double *get_float(size_t i, size_t j) {
		return &data_[i * cols_ + j];
	}
	const double *get_float(size_t i, size_t j) const {
		return &data_[i * cols_ + j];
	}
	uint32_t *get_fixed(size_t i, size_t j) {
		return &fixed_[i * cols_ + j];
	}
	const uint32_t *get_fixed(size_t i, size_t j) const {
		return &fixed_[i * cols_ + j];
	}

	size_t rows() const {
		return rows_;
	}
	size_t cols() const {
		return cols_;
	}

	void fallback() {
		if (max_steps_ == 0) {
			return;
		}

		const size_t stride = 256 / 8 / sizeof(double);
		const size_t S = rows_ * cols_;
		const size_t T = S % stride;

		const __m256d dequant = _mm256_set1_pd(dequant_);
		const __m256d offset = _mm256_set1_pd(min_);

		// Alignment guaranteed from alloc
		for (size_t i = 0; i < S - T; i += stride) {
			__m128i orig = _mm_load_si128((__m128i *)(fixed_ + i));
			__m256d cvt = _mm256_cvtepi32_pd(orig);
			__m256d final = _mm256_fmadd_pd(cvt, dequant, offset);
			_mm256_store_pd(data_ + i, final);
		}
		for (size_t i = S - T; i < S; i++) {
			data_[i] = double(fixed_[i]) * dequant_ + min_;
		}
	}

	void promote() {
		const size_t stride = 256 / 8 / sizeof(double);
		const size_t S = rows_ * cols_;
		const size_t T = S % stride;

		double range = max_ - min_;
		if (std::fpclassify(range) != FP_NORMAL) {
			bad_ = true;
			return;
		}

		quant_ = std::floor((double(0xfffffff8U) / 4) / range);
		dequant_ = 1.0 / quant_;

		const double ERR_GROWTH_RATE = 4e11; // Empirically found constant
		max_steps_ = std::fmin(std::floor(quant_ * quant_ / ERR_GROWTH_RATE), 1e18);

		const __m256d quant = _mm256_set1_pd(quant_);
		const __m256d offset = _mm256_set1_pd(min_);

		// Alignment guaranteed from alloc
		for (size_t i = 0; i < S - T; i += stride) {
			__m256d orig = _mm256_load_pd(data_ + i);
			__m256d scaled = _mm256_mul_pd(quant, _mm256_sub_pd(orig, offset));
			__m128i cvt = _mm256_cvtpd_epi32(scaled);
			_mm_store_si128((__m128i *)(fixed_ + i), cvt);
		}
		for (size_t i = S - T; i < S; i++) {
			fixed_[i] = std::nearbyint((data_[i] - min_) * quant_);
		}
	}

	// Returns true if the fixed point representation is valid
	bool prerun() {
		if (__builtin_expect(bad_, false))
			return false;

		if (max_steps_ == 0) {
			promote();
			if (bad_)
				return false;
			num_steps_ = 0;
		}

		if (num_steps_++ >= max_steps_) {
			fallback();
			bad_ = true;
			return false;
		} else {
			return true;
		}
	}
};

Proxy::operator double() const {
	if (g->bad_ || g->num_steps_ >= g->max_steps_)
		return g->data_[idx];

	return g->fixed_[idx] * g->dequant_ + g->min_;
}

Proxy Proxy::operator=(double x) {
	// Fallback to floats only if we get something weird
	if (g->bad_ || !std::isfinite(x)) {
		g->bad_ = true;
		g->max_steps_ = 0;
		g->data_[idx] = x;
		return *this;
	}

	if (g->num_steps_ != 0 && g->max_steps_ != 0) {
		g->fallback();
		g->bad_ = true;
		g->data_[idx] = x;
		g->max_steps_ = 0;
		g->num_steps_ = 0;
		return *this;
	}

	g->min_ = std::min(g->min_, x);
	g->max_ = std::max(g->max_, x);
	g->data_[idx] = x;
	g->max_steps_ = 0;
	g->num_steps_ = 0;

	return *this;
}

template <bool aligned> static void _apply_stencil(const Grid &old_grid, Grid &new_grid, size_t start, size_t end) {
	const size_t N = old_grid.rows();
	const size_t M = old_grid.cols();

	const double *__restrict__ old_data = old_grid.get_float(0, 0);
	double *__restrict__ new_data = new_grid.get_float(0, 0);

	if constexpr (!aligned) {
		if (M <= 2) {
			if (start < end)
				memcpy(new_data + start * M, old_data + start * M, (end - start) * M * sizeof(double));
			return;
		}
	}

	const size_t stride = 256 / 8 / sizeof(double);

	const __m256d two = _mm256_set1_pd(0.5);
	const __m256d eight = _mm256_set1_pd(0.125);

	for (int i = start; i < end; i++) {
		const double *row = old_data + (size_t)i * M;
		double *nrow = new_data + (size_t)i * M;

		if constexpr (aligned) {
			for (int j = 0; j < M; j += stride) {
				__m256d up = _mm256_load_pd(row + j - M);
				__m256d down = _mm256_load_pd(row + j + M);
				__m256d left = _mm256_loadu_pd(row + j - 1);
				__m256d right = _mm256_loadu_pd(row + j + 1);
				__m256d cur = _mm256_load_pd(row + j);

				__m256d ver = _mm256_add_pd(up, down);
				__m256d hor = _mm256_add_pd(left, right);

				__m256d around = _mm256_mul_pd(_mm256_add_pd(ver, hor), eight);
				cur = _mm256_fmadd_pd(two, cur, around);

				_mm256_store_pd(nrow + j, cur);
			}
		} else {
			const size_t T = (M - 1) % stride;
			for (int j = 1; j < M - T; j += stride) {
				__m256d up = _mm256_loadu_pd(row + j - M);
				__m256d down = _mm256_loadu_pd(row + j + M);
				__m256d left = _mm256_loadu_pd(row + j - 1);
				__m256d right = _mm256_loadu_pd(row + j + 1);
				__m256d cur = _mm256_loadu_pd(row + j);

				__m256d ver = _mm256_add_pd(up, down);
				__m256d hor = _mm256_add_pd(left, right);

				__m256d around = _mm256_mul_pd(_mm256_add_pd(ver, hor), eight);
				cur = _mm256_fmadd_pd(two, cur, around);

				_mm256_storeu_pd(nrow + j, cur);
			}
			for (int j = M - T; j < M; j++)
				nrow[j] = 0.5 * row[j] + 0.125 * (row[j - M] + row[j - 1] + row[j + M] + row[j + 1]);
		}

		nrow[M - 1] = row[M - 1];
		nrow[0] = row[0];
	}

	if (start < end) {
		new_data[start * M] = old_data[start * M];
		new_data[(end - 1) * M + M - 1] = old_data[(end - 1) * M + M - 1];
	}
}

struct ThreadInfo {
	union {
		struct {
			const Grid *old_grid;
			Grid *new_grid;
			size_t start, end;
			bool aligned;
		};
		uint8_t pad[64];
	};
};

static constexpr int NTHREADS = 4;
alignas(64) static ThreadInfo tis[NTHREADS - 1];
alignas(64) static std::atomic_int done = 0;
alignas(64) static std::atomic_int epoch = 0;

static void worker(int idx) {
	const ThreadInfo &t = tis[idx];
	int cur_epoch = 0;

	while (true) {
		cur_epoch++;
		while (epoch.load(std::memory_order_acquire) < cur_epoch)
			_mm_pause();

		if (__builtin_expect(t.aligned, true))
			_apply_stencil<true>(*t.old_grid, *t.new_grid, t.start, t.end);
		else
			_apply_stencil<false>(*t.old_grid, *t.new_grid, t.start, t.end);

		done.fetch_add(1, std::memory_order_release);
	}
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
static void apply_stencil(const Grid &old_grid, Grid &new_grid) {
	const size_t N = old_grid.rows();
	const size_t M = old_grid.cols();
	const bool aligned = (M % (256 / 64)) == 0;

	if (N >= 8) {
		size_t base = 1;
		const size_t stride = (N - 2) / NTHREADS;

		for (int i = 0; i < NTHREADS - 1; i++) {
			tis[i] = {&old_grid, &new_grid, base, base + stride, aligned};
			base += stride;
		}

		done.store(0, std::memory_order_relaxed);
		epoch.fetch_add(1, std::memory_order_release);

		if (__builtin_expect(aligned, true))
			_apply_stencil<true>(old_grid, new_grid, base, N - 1);
		else
			_apply_stencil<false>(old_grid, new_grid, base, N - 1);

		std::memcpy(new_grid.get_float(N - 1, 0), old_grid.get_float(N - 1, 0), M * sizeof(double));
		std::memcpy(new_grid.get_float(0, 0), old_grid.get_float(0, 0), M * sizeof(double));

		while (done.load(std::memory_order_acquire) < NTHREADS - 1)
			_mm_pause();
	} else {
		_apply_stencil<false>(old_grid, new_grid, 1, N - 1);

		std::memcpy(new_grid.get_float(N - 1, 0), old_grid.get_float(N - 1, 0), M * sizeof(double));
		std::memcpy(new_grid.get_float(0, 0), old_grid.get_float(0, 0), M * sizeof(double));
	}
}

__attribute__((constructor)) static void init_workers() {
	for (int i = 0; i < NTHREADS - 1; i++) {
		std::thread(worker, i).detach();
	}
}
