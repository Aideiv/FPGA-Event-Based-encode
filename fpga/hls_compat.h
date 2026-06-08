// hls_compat.h — HLS compatibility layer for standalone simulation
//
// When compiled for simulation (g++ -D__SIMULATION__), this provides
// minimal equivalents of Vitis HLS types (ap_fixed, ap_uint, hls::stream, etc.)
// so the FPGA modules can compile on a standard Linux CI runner without
// the full Xilinx/Vitis HLS toolchain.
//
// When compiled under Vitis HLS (no __SIMULATION__ defined), this includes
// the real HLS headers instead.
#pragma once

#ifdef __SIMULATION__
// ===========================================================================
// Standalone simulation types
// ===========================================================================
#include <cstdint>
#include <vector>
#include <cmath>

// Simple ap_int for fixed-size signed integers
template<int W>
class ap_int {
public:
    int64_t val;
    ap_int() : val(0) {}
    ap_int(int64_t v) : val(v & ((1LL << W) - 1)) {
        // Sign-extend
        if (v & (1LL << (W - 1))) val |= ~((1LL << W) - 1);
    }
    ap_int(int v) : ap_int(static_cast<int64_t>(v)) {}
    ap_int(unsigned long long v) : ap_int(static_cast<int64_t>(v)) {}

    operator int64_t() const { return val; }
    operator int() const { return static_cast<int>(val); }
    operator bool() const { return val != 0; }

    ap_int operator+(const ap_int& o) const { return ap_int(val + o.val); }
    ap_int operator-(const ap_int& o) const { return ap_int(val - o.val); }
    ap_int operator*(const ap_int& o) const { return ap_int(val * o.val); }
    ap_int operator/(const ap_int& o) const { return o.val ? ap_int(val / o.val) : ap_int(0); }
    ap_int operator-() const { return ap_int(-val); }
    ap_int& operator+=(const ap_int& o) { val += o.val; return *this; }
    ap_int& operator=(int64_t v) { val = v & ((1LL << W) - 1); return *this; }

    bool operator>(const ap_int& o) const { return val > o.val; }
    bool operator<(const ap_int& o) const { return val < o.val; }
    bool operator>=(const ap_int& o) const { return val >= o.val; }
    bool operator<=(const ap_int& o) const { return val <= o.val; }
    bool operator==(const ap_int& o) const { return val == o.val; }
    bool operator!=(const ap_int& o) const { return val != o.val; }
};

// Simple ap_uint for fixed-size unsigned integers
template<int W>
class ap_uint {
public:
    uint64_t val;
    ap_uint() : val(0) {}
    ap_uint(uint64_t v) : val(v & ((1ULL << W) - 1)) {}
    ap_uint(int v) : val(static_cast<uint64_t>(v) & ((1ULL << W) - 1)) {}
    ap_uint(unsigned long long v) : val(v & ((1ULL << W) - 1)) {}

    operator uint64_t() const { return val; }
    operator int() const { return static_cast<int>(val); }
    operator bool() const { return val != 0; }

    ap_uint operator+(const ap_uint& o) const { return ap_uint(val + o.val); }
    ap_uint operator-(const ap_uint& o) const { return ap_uint(val - o.val); }
    ap_uint operator*(const ap_uint& o) const { return ap_uint(val * o.val); }
    ap_uint operator/(const ap_uint& o) const { return o.val ? ap_uint(val / o.val) : ap_uint(0); }
    ap_uint operator&(const ap_uint& o) const { return ap_uint(val & o.val); }
    ap_uint operator|(const ap_uint& o) const { return ap_uint(val | o.val); }
    ap_uint operator<<(int n) const { return ap_uint(val << n); }
    ap_uint operator>>(int n) const { return ap_uint(val >> n); }

    ap_uint& operator++() { val = (val + 1) & ((1ULL << W) - 1); return *this; }
    ap_uint& operator=(uint64_t v) { val = v & ((1ULL << W) - 1); return *this; }

    bool operator>(const ap_uint& o) const { return val > o.val; }
    bool operator<(const ap_uint& o) const { return val < o.val; }
    bool operator>=(const ap_uint& o) const { return val >= o.val; }
    bool operator<=(const ap_uint& o) const { return val <= o.val; }
    bool operator==(const ap_uint& o) const { return val == o.val; }
    bool operator!=(const ap_uint& o) const { return val != o.val; }
};

// ap_fixed<W,I> — fixed-point with W total bits and I integer bits
template<int W, int I>
class ap_fixed {
public:
    static const int FRAC_BITS = W - I;
    static const int64_t SCALE = 1LL << FRAC_BITS;
    int64_t val;

    ap_fixed() : val(0) {}
    ap_fixed(float f) {
        float scaled = f * SCALE;
        int64_t max_val = (1LL << (W - 1)) - 1;
        int64_t min_val = -(1LL << (W - 1));
        if (scaled > max_val) scaled = (float)max_val;
        if (scaled < min_val) scaled = (float)min_val;
        val = (int64_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
    }
    ap_fixed(double d) : ap_fixed((float)d) {}
    ap_fixed(int v) : ap_fixed((float)v) {}

    operator float() const { return (float)val / SCALE; }
    operator double() const { return (double)val / SCALE; }

    ap_fixed operator+(const ap_fixed& o) const { ap_fixed r; r.val = val + o.val; return r; }
    ap_fixed operator-(const ap_fixed& o) const { ap_fixed r; r.val = val - o.val; return r; }
    ap_fixed operator*(const ap_fixed& o) const {
        ap_fixed r;
        r.val = (val * o.val) >> FRAC_BITS;
        return r;
    }
    ap_fixed operator/(const ap_fixed& o) const {
        if (o.val == 0) return ap_fixed(0.0f);
        ap_fixed r;
        r.val = (val << FRAC_BITS) / o.val;
        return r;
    }
    ap_fixed operator-() const { ap_fixed r; r.val = -val; return r; }
    ap_fixed& operator+=(const ap_fixed& o) { val += o.val; return *this; }

    bool operator>(const ap_fixed& o) const { return val > o.val; }
    bool operator<(const ap_fixed& o) const { return val < o.val; }
    bool operator==(const ap_fixed& o) const { return val == o.val; }
};

// ap_ufixed<W,I> — unsigned fixed-point
template<int W, int I>
class ap_ufixed {
public:
    static const int FRAC_BITS = W - I;
    static const uint64_t SCALE = 1ULL << FRAC_BITS;
    uint64_t val;

    ap_ufixed() : val(0) {}
    ap_ufixed(float f) {
        float scaled = f * SCALE;
        uint64_t max_val = (1ULL << W) - 1;
        if (scaled < 0) scaled = 0;
        if (scaled > max_val) scaled = (float)max_val;
        val = (uint64_t)(scaled + 0.5f);
    }
    ap_ufixed(double d) : ap_ufixed((float)d) {}
    ap_ufixed(int v) : ap_ufixed((float)v) {}
    ap_ufixed(uint64_t v) : val(v & ((1ULL << W) - 1)) {}

    operator float() const { return (float)val / SCALE; }
    operator double() const { return (double)val / SCALE; }

    ap_ufixed operator+(const ap_ufixed& o) const { ap_ufixed r; r.val = val + o.val; return r; }
    ap_ufixed operator-(const ap_ufixed& o) const { ap_ufixed r; r.val = val - o.val; return r; }
    ap_ufixed operator*(const ap_ufixed& o) const {
        ap_ufixed r;
        r.val = (val * o.val) >> FRAC_BITS;
        return r;
    }
    ap_ufixed operator/(const ap_ufixed& o) const {
        if (o.val == 0) return ap_ufixed(0.0f);
        ap_ufixed r;
        r.val = (val << FRAC_BITS) / o.val;
        return r;
    }

    bool operator>(const ap_ufixed& o) const { return val > o.val; }
    bool operator<(const ap_ufixed& o) const { return val < o.val; }
    bool operator==(const ap_ufixed& o) const { return val == o.val; }
};

// hls::stream
namespace hls {
    template<typename T>
    class stream {
        std::vector<T> buf;
    public:
        stream(const char*) {}
        bool empty() const { return buf.empty(); }
        void write(const T& v) { buf.push_back(v); }
        void read(T& v) { v = buf.back(); buf.pop_back(); }
        T read() { T v = buf.back(); buf.pop_back(); return v; }
    };

    template<typename T>
    T min(T a, T b) { return (a < b) ? a : b; }
}

// AXI Stream data type
template<int W, int D, int U, int I>
struct ap_axiu {
    ap_uint<W> data;
    ap_uint<D / 8> keep;
    ap_uint<U> user;
    ap_uint<1> last;
    ap_uint<I> id;
    ap_uint<D / 8> dest;
};

// HLS pragmas are no-ops in simulation
#define PRAGMA_HLS(x)
#define PRAGMA_RESET(x)
#define HLS_INLINE

#else
// ===========================================================================
// Real synthesis: include actual Vitis HLS headers
// ===========================================================================
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_axi_sdata.h>
#endif