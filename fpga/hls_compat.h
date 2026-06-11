// hls_compat.h — Vitis HLS compatibility for g++ (std=c++11)
#pragma once
#ifdef __SIMULATION__
#include <cstdint>
#include <vector>
#include <cmath>

template <int W>
class ap_uint;
template <int W>
class ap_int;
template <int W, int I>
class ap_fixed;
template <int W, int I>
class ap_ufixed;

// ===========================================================================
// ap_uint<W>
// ===========================================================================
template <int W>
class ap_uint {
    static constexpr uint64_t MASK = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);

   public:
    uint64_t val;
    ap_uint() : val(0) {}
    ap_uint(uint64_t v) : val(v & MASK) {}
    ap_uint(int v) : val(static_cast<uint64_t>(v) & MASK) {}
    ap_uint(unsigned int v) : val(static_cast<uint64_t>(v) & MASK) {}
    ap_uint(float f) : val(static_cast<uint64_t>(f < 0 ? 0 : f) & MASK) {}
    ap_uint(double d) : val(static_cast<uint64_t>(d < 0 ? 0 : d) & MASK) {}
    template <int W2>
    ap_uint(const ap_uint<W2>& o) : val(o.val & MASK) {}
    template <int W2>
    ap_uint(const ap_int<W2>& o) : val(static_cast<uint64_t>(o.val) & MASK) {}

    uint64_t to_uint64() const { return val; }
    // Implicit integral conversion (matches Vitis ap_uint): arithmetic,
    // comparisons, array indexing and static_cast all go through the
    // builtin operators on uint64_t — width masking happens on
    // construction/assignment. No member arithmetic/comparison operators:
    // they would be ambiguous against the builtin ones.
    operator uint64_t() const { return val; }

    // bit-slice: x(hi, lo) — hi/lo are runtime values, width capped at W
    ap_uint<(W > 0 ? W : 1)> operator()(int hi, int lo) const {
        const int w = hi - lo + 1;
        const uint64_t mask = (w >= 64) ? ~0ULL : ((1ULL << (w > 0 ? w : 1)) - 1);
        return ap_uint<(W > 0 ? W : 1)>((val >> lo) & mask);
    }

    ap_uint& operator++() {
        val = (val + 1) & MASK;
        return *this;
    }
    ap_uint operator++(int) {
        ap_uint t(*this);
        val = (val + 1) & MASK;
        return t;
    }
    ap_uint& operator--() {
        val = (val - 1) & MASK;
        return *this;
    }
    ap_uint operator--(int) {
        ap_uint t(*this);
        val = (val - 1) & MASK;
        return t;
    }
    ap_uint& operator+=(uint64_t v) {
        val = (val + v) & MASK;
        return *this;
    }
    ap_uint& operator-=(uint64_t v) {
        val = (val - v) & MASK;
        return *this;
    }
    ap_uint& operator=(uint64_t v) {
        val = v & MASK;
        return *this;
    }
    template <int W2>
    ap_uint& operator=(const ap_uint<W2>& o) {
        val = o.val & MASK;
        return *this;
    }
};

// ===========================================================================
// ap_int<W>
// ===========================================================================
template <int W>
class ap_int {
    static constexpr int64_t MN = -(1LL << (W - 1));
    static constexpr int64_t MX = (1LL << (W - 1)) - 1;
    static int64_t clamp(int64_t v) { return v > MX ? MX : (v < MN ? MN : v); }

   public:
    int64_t val;
    ap_int() : val(0) {}
    ap_int(int64_t v) : val(clamp(v)) {}
    ap_int(int v) : ap_int(static_cast<int64_t>(v)) {}
    template <int W2>
    ap_int(const ap_int<W2>& o) : val(clamp(o.val)) {}
    template <int W2>
    ap_int(const ap_uint<W2>& o) : val(clamp(static_cast<int64_t>(o.val))) {}
    template <int W2, int I2>
    ap_int(const ap_fixed<W2, I2>& f) : ap_int(static_cast<int64_t>(static_cast<float>(f))) {}
    // Implicit integral conversion — see ap_uint: builtin operators do the
    // arithmetic/comparisons, clamping happens on construction/assignment.
    operator int64_t() const { return val; }

    // Allow sclk_prev = spi_sclk (ap_int = ap_uint)
    template <int W2>
    ap_int& operator=(const ap_uint<W2>& o) {
        val = clamp(static_cast<int64_t>(o.val));
        return *this;
    }
    ap_int& operator=(int64_t v) {
        val = clamp(v);
        return *this;
    }

    ap_int& operator++() {
        val = clamp(val + 1);
        return *this;
    }
    ap_int operator++(int) {
        ap_int t(*this);
        val = clamp(val + 1);
        return t;
    }
    ap_int& operator--() {
        val = clamp(val - 1);
        return *this;
    }
    ap_int operator--(int) {
        ap_int t(*this);
        val = clamp(val - 1);
        return t;
    }
    ap_int& operator+=(int64_t v) {
        val = clamp(val + v);
        return *this;
    }
    ap_int& operator-=(int64_t v) {
        val = clamp(val - v);
        return *this;
    }
};

// ===========================================================================
// ap_fixed<W,I>
// ===========================================================================
template <int W, int I>
class ap_fixed {
    static const int FRAC = W - I;
    static constexpr int64_t SCALE = 1LL << FRAC;
    static constexpr int64_t MX = (1LL << (W - 1)) - 1;
    static constexpr int64_t MN = -(1LL << (W - 1));
    int64_t val_;
    static int64_t sat(int64_t v) { return v > MX ? MX : (v < MN ? MN : v); }

   public:
    ap_fixed() : val_(0) {}
    ap_fixed(float f) : val_(sat(static_cast<int64_t>(f * SCALE + (f >= 0 ? 0.5f : -0.5f)))) {}
    ap_fixed(double d) : ap_fixed(static_cast<float>(d)) {}
    ap_fixed(int v) : ap_fixed(static_cast<float>(v)) {}
    template <int W2, int I2>
    ap_fixed(const ap_fixed<W2, I2>& o) : ap_fixed(static_cast<float>(o)) {}
    template <int W2, int I2>
    ap_fixed(const ap_ufixed<W2, I2>& o) : ap_fixed(static_cast<float>(o)) {}
    template <int W2>
    ap_fixed(const ap_uint<W2>& o) : ap_fixed(static_cast<float>(o.to_uint64())) {}
    operator float() const { return static_cast<float>(val_) / SCALE; }
    // explicit: an implicit double conversion alongside float makes
    // converting-constructor overload resolution ambiguous
    explicit operator double() const { return static_cast<double>(val_) / SCALE; }
    ap_fixed operator+(ap_fixed o) const {
        ap_fixed r;
        r.val_ = sat(val_ + o.val_);
        return r;
    }
    ap_fixed operator-(ap_fixed o) const {
        ap_fixed r;
        r.val_ = sat(val_ - o.val_);
        return r;
    }
    ap_fixed operator*(ap_fixed o) const {
        ap_fixed r;
        r.val_ = sat((val_ * o.val_) >> FRAC);
        return r;
    }
    ap_fixed operator/(ap_fixed o) const {
        if (o.val_ == 0) return ap_fixed(0);
        ap_fixed r;
        r.val_ = sat((val_ << FRAC) / o.val_);
        return r;
    }
    ap_fixed operator-() const {
        ap_fixed r;
        r.val_ = -val_;
        return r;
    }
    ap_fixed& operator+=(ap_fixed o) {
        val_ = sat(val_ + o.val_);
        return *this;
    }
    ap_fixed& operator-=(ap_fixed o) {
        val_ = sat(val_ - o.val_);
        return *this;
    }
    ap_fixed operator>>(int n) const {
        ap_fixed r;
        r.val_ = sat(val_ >> n);
        return r;
    }
    ap_fixed operator<<(int n) const {
        ap_fixed r;
        r.val_ = sat(val_ << n);
        return r;
    }
    // Comparisons intentionally omitted: they resolve through the implicit
    // float conversion (member overloads would be ambiguous against it
    // when comparing with int/float literals).
};

// ===========================================================================
// ap_ufixed<W,I>
// ===========================================================================
template <int W, int I>
class ap_ufixed {
    static const int FRAC = W - I;
    static constexpr uint64_t SCALE = 1ULL << FRAC;
    static constexpr uint64_t MASK = (W >= 64) ? ~0ULL : ((1ULL << W) - 1);
    uint64_t val_;

   public:
    ap_ufixed() : val_(0) {}
    ap_ufixed(float f) {
        float s = f * SCALE;
        if (s < 0) s = 0;
        if (s > MASK) s = MASK;
        val_ = static_cast<uint64_t>(s + 0.5f) & MASK;
    }
    ap_ufixed(double d) : ap_ufixed(static_cast<float>(d)) {}
    ap_ufixed(int v) : ap_ufixed(static_cast<float>(v)) {}
    explicit ap_ufixed(uint64_t v) : val_(v & MASK) {}
    // ap_uint → ap_ufixed (aer_interface, ring_buffer)
    template <int W2>
    explicit ap_ufixed(const ap_uint<W2>& v) : ap_ufixed(static_cast<float>(v.val)) {}
    template <int W2, int I2>
    ap_ufixed(const ap_ufixed<W2, I2>& o) : ap_ufixed(static_cast<float>(o)) {}
    operator float() const { return static_cast<float>(val_) / SCALE; }
    // explicit: see ap_fixed — keeps width-converting construction unambiguous
    explicit operator double() const { return static_cast<double>(val_) / SCALE; }
    ap_ufixed operator+(ap_ufixed o) const {
        ap_ufixed r;
        r.val_ = (val_ + o.val_) & MASK;
        return r;
    }
    ap_ufixed operator-(ap_ufixed o) const {
        ap_ufixed r;
        r.val_ = (val_ - o.val_) & MASK;
        return r;
    }
    ap_ufixed operator*(ap_ufixed o) const {
        ap_ufixed r;
        r.val_ = ((val_ * o.val_) >> FRAC) & MASK;
        return r;
    }
    ap_ufixed operator/(ap_ufixed o) const {
        if (o.val_ == 0) return ap_ufixed(0);
        ap_ufixed r;
        r.val_ = ((val_ << FRAC) / o.val_) & MASK;
        return r;
    }
    bool operator>(ap_ufixed o) const { return val_ > o.val_; }
    bool operator<(ap_ufixed o) const { return val_ < o.val_; }
    bool operator>=(ap_ufixed o) const { return val_ >= o.val_; }
    bool operator<=(ap_ufixed o) const { return val_ <= o.val_; }
    bool operator==(ap_ufixed o) const { return val_ == o.val_; }
    bool operator!=(ap_ufixed o) const { return val_ != o.val_; }
};

// ===========================================================================
// hls::stream (FIFO)
// ===========================================================================
namespace hls {
template <typename T>
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
template <typename T>
T min(T a, T b) {
    return (a < b) ? a : b;
}
}  // namespace hls

template <int W, int D, int U, int I>
struct ap_axiu {
    ap_uint<W> data;
    ap_uint<D / 8> keep;
    ap_uint<U> user;
    ap_uint<1> last;
    ap_uint<I> id;
    ap_uint<D / 8> dest;
};

#define PRAGMA_HLS(x)
#define PRAGMA_RESET(x)
#define HLS_INLINE

#else
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_axi_sdata.h>
#endif