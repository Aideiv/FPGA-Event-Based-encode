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
#include <type_traits>

// ===========================================================================
// ap_uint<W> — fixed-width unsigned integer
// ===========================================================================
template<int W>
class ap_uint {
    static constexpr uint64_t MASK = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);

public:
    uint64_t val;   // Vitis HLS compatible public member

    ap_uint() : val(0) {}
    ap_uint(uint64_t v) : val(v & MASK) {}
    ap_uint(int v) : val(static_cast<uint64_t>(v) & MASK) {}

    uint64_t to_uint64() const { return val; }
    operator float() const { return static_cast<float>(val); }
    operator double() const { return static_cast<double>(val); }
    operator int() const { return static_cast<int>(val); }
    operator unsigned int() const { return static_cast<unsigned int>(val); }

    // Vitis HLS bit-slice operator: x(hi, lo) → bits[hi:lo]
    ap_uint<(W>0?W:1)> operator()(int hi, int lo) const {
        const int WIDTH = hi - lo + 1;
        return ap_uint<(WIDTH>0?WIDTH:1)>((val >> lo) & ((1ULL << WIDTH) - 1));
    }

    // Same-type arithmetic
    ap_uint operator+(ap_uint o) const { return ap_uint(val + o.val); }
    ap_uint operator-(ap_uint o) const { return ap_uint(val - o.val); }
    ap_uint operator*(ap_uint o) const { return ap_uint(val * o.val); }
    ap_uint operator/(ap_uint o) const { return o.val ? ap_uint(val / o.val) : ap_uint(0); }
    ap_uint operator%(ap_uint o) const { return o.val ? ap_uint(val % o.val) : ap_uint(0); }
    ap_uint operator&(ap_uint o) const { return ap_uint(val & o.val); }
    ap_uint operator|(ap_uint o) const { return ap_uint(val | o.val); }
    ap_uint operator<<(int n) const { return ap_uint(val << n); }
    ap_uint operator>>(int n) const { return ap_uint(val >> n); }

    ap_uint& operator++() { val = (val + 1) & MASK; return *this; }
    ap_uint& operator=(uint64_t v) { val = v & MASK; return *this; }

    // Same-type comparison
    bool operator>(ap_uint o)  const { return val > o.val; }
    bool operator<(ap_uint o)  const { return val < o.val; }
    bool operator>=(ap_uint o) const { return val >= o.val; }
    bool operator<=(ap_uint o) const { return val <= o.val; }
    bool operator==(ap_uint o) const { return val == o.val; }
    bool operator!=(ap_uint o) const { return val != o.val; }
};

// ===========================================================================
// ap_int<W> — fixed-width signed integer
// ===========================================================================
template<int W>
class ap_int {
    static constexpr uint64_t UMAX = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);
    static constexpr int64_t MIN_VAL = -(1LL << (W - 1));
    static constexpr int64_t MAX_VAL = (1LL << (W - 1)) - 1;
    int64_t val_;

    static int64_t clamp(int64_t v) {
        if (v > MAX_VAL) return MAX_VAL;
        if (v < MIN_VAL) return MIN_VAL;
        return v;
    }

public:
    ap_int() : val_(0) {}
    explicit ap_int(int64_t v) : val_(clamp(v)) {}
    ap_int(int v) : ap_int(static_cast<int64_t>(v)) {}

    operator int64_t() const { return val_; }
    operator int() const { return static_cast<int>(val_); }

    ap_int operator+(ap_int o) const { return ap_int(val_ + o.val_); }
    ap_int operator-(ap_int o) const { return ap_int(val_ - o.val_); }
    ap_int operator*(ap_int o) const { return ap_int(val_ * o.val_); }
    ap_int operator/(ap_int o) const { return o.val_ ? ap_int(val_ / o.val_) : ap_int(0); }
    ap_int operator-() const { return ap_int(-val_); }
    ap_int& operator+=(ap_int o) { val_ = clamp(val_ + o.val_); return *this; }
    ap_int& operator=(int64_t v) { val_ = clamp(v); return *this; }

    bool operator>(ap_int o)  const { return val_ > o.val_; }
    bool operator<(ap_int o)  const { return val_ < o.val_; }
    bool operator>=(ap_int o) const { return val_ >= o.val_; }
    bool operator<=(ap_int o) const { return val_ <= o.val_; }
    bool operator==(ap_int o) const { return val_ == o.val_; }
    bool operator!=(ap_int o) const { return val_ != o.val_; }
};

// ===========================================================================
// ap_fixed<W,I> — signed fixed-point
// ===========================================================================
template<int W, int I>
class ap_fixed {
    static const int FRAC = W - I;
    static constexpr int64_t SCALE = 1LL << FRAC;
    static constexpr int64_t MAX_VAL = (1LL << (W - 1)) - 1;
    static constexpr int64_t MIN_VAL = -(1LL << (W - 1));
    int64_t val_;

    static int64_t saturate(int64_t v) {
        if (v > MAX_VAL) return MAX_VAL;
        if (v < MIN_VAL) return MIN_VAL;
        return v;
    }

public:
    ap_fixed() : val_(0) {}
    ap_fixed(float f)  : val_(saturate(static_cast<int64_t>(f * SCALE + (f >= 0 ? 0.5f : -0.5f)))) {}
    ap_fixed(double d) : ap_fixed(static_cast<float>(d)) {}
    ap_fixed(int v)    : ap_fixed(static_cast<float>(v)) {}

    operator float()  const { return static_cast<float>(val_) / SCALE; }
    operator double() const { return static_cast<double>(val_) / SCALE; }

    ap_fixed operator+(ap_fixed o) const { ap_fixed r; r.val_ = saturate(val_ + o.val_); return r; }
    ap_fixed operator-(ap_fixed o) const { ap_fixed r; r.val_ = saturate(val_ - o.val_); return r; }
    ap_fixed operator*(ap_fixed o) const {
        ap_fixed r;
        r.val_ = saturate((val_ * o.val_) >> FRAC);
        return r;
    }
    ap_fixed operator/(ap_fixed o) const {
        if (o.val_ == 0) return ap_fixed(0.0f);
        ap_fixed r;
        r.val_ = saturate((val_ << FRAC) / o.val_);
        return r;
    }
    ap_fixed operator-() const { ap_fixed r; r.val_ = -val_; return r; }
    ap_fixed& operator+=(ap_fixed o) { val_ = saturate(val_ + o.val_); return *this; }

    // Right-shift by integer (Vitis HLS: arithmetic shift for signed)
    ap_fixed operator>>(int n) const {
        ap_fixed r;
        r.val_ = saturate(val_ >> n);
        return r;
    }
    ap_fixed operator<<(int n) const {
        ap_fixed r;
        r.val_ = saturate(val_ << n);
        return r;
    }

    bool operator>(ap_fixed o)  const { return val_ > o.val_; }
    bool operator<(ap_fixed o)  const { return val_ < o.val_; }
    bool operator>=(ap_fixed o) const { return val_ >= o.val_; }
    bool operator<=(ap_fixed o) const { return val_ <= o.val_; }
    bool operator==(ap_fixed o) const { return val_ == o.val_; }
    bool operator!=(ap_fixed o) const { return val_ != o.val_; }
};

// ===========================================================================
// ap_ufixed<W,I> — unsigned fixed-point
// ===========================================================================
template<int W, int I>
class ap_ufixed {
    static const int FRAC = W - I;
    static constexpr uint64_t SCALE = 1ULL << FRAC;
    static constexpr uint64_t MASK = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);
    uint64_t val_;

public:
    ap_ufixed() : val_(0) {}
    ap_ufixed(float f) {
        float scaled = f * SCALE;
        if (scaled < 0) scaled = 0;
        if (scaled > MASK) scaled = static_cast<float>(MASK);
        val_ = static_cast<uint64_t>(scaled + 0.5f) & MASK;
    }
    ap_ufixed(double d) : ap_ufixed(static_cast<float>(d)) {}
    ap_ufixed(int v)    : ap_ufixed(static_cast<float>(v)) {}
    explicit ap_ufixed(uint64_t v) : val_(v & MASK) {}

    operator float()  const { return static_cast<float>(val_) / SCALE; }
    operator double() const { return static_cast<double>(val_) / SCALE; }

    ap_ufixed operator+(ap_ufixed o) const { ap_ufixed r; r.val_ = (val_ + o.val_) & MASK; return r; }
    ap_ufixed operator-(ap_ufixed o) const { ap_ufixed r; r.val_ = (val_ - o.val_) & MASK; return r; }
    ap_ufixed operator*(ap_ufixed o) const {
        ap_ufixed r;
        r.val_ = ((val_ * o.val_) >> FRAC) & MASK;
        return r;
    }
    ap_ufixed operator/(ap_ufixed o) const {
        if (o.val_ == 0) return ap_ufixed(0.0f);
        ap_ufixed r;
        r.val_ = ((val_ << FRAC) / o.val_) & MASK;
        return r;
    }

    bool operator>(ap_ufixed o)  const { return val_ > o.val_; }
    bool operator<(ap_ufixed o)  const { return val_ < o.val_; }
    bool operator>=(ap_ufixed o) const { return val_ >= o.val_; }
    bool operator<=(ap_ufixed o) const { return val_ <= o.val_; }
    bool operator==(ap_ufixed o) const { return val_ == o.val_; }
    bool operator!=(ap_ufixed o) const { return val_ != o.val_; }
};

// ===========================================================================
// hls::stream
// ===========================================================================
namespace hls {
    template<typename T>
    class stream {
        std::vector<T> buf_;
    public:
        stream(const char*) {}
        bool empty() const { return buf_.empty(); }
        void write(const T& v) { buf_.push_back(v); }
        void read(T& v) {
            v = buf_.front();
            buf_.erase(buf_.begin());
        }
        T read() {
            T v = buf_.front();
            buf_.erase(buf_.begin());
            return v;
        }
    };

    template<typename T>
    T min(T a, T b) { return (a < b) ? a : b; }
}

// ===========================================================================
// ap_axiu — AXI4-Stream data type (synthesis only stub)
// ===========================================================================
template<int W, int D, int U, int I>
struct ap_axiu {
    ap_uint<W> data;
    ap_uint<D / 8> keep;
    ap_uint<U> user;
    ap_uint<1> last;
    ap_uint<I> id;
    ap_uint<D / 8> dest;
};

// ===========================================================================
// HLS pragma stubs
// ===========================================================================
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