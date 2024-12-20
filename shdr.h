#ifndef SHDR_SHDR_H
#define SHDR_SHDR_H

#pragma once

// GLSL reference: https://registry.khronos.org/OpenGL/specs/es/3.2/GLSL_ES_Specification_3.20.html

#include <ostream>
#include <tuple>
#include <utility>
#include <concepts>
#include <functional>

#include <cmath>

namespace gles {
#ifdef OPENGL_ES_32
#include <GLES3/gl32.h>
#else

#include <GLES3/gl31.h>

#define GL_DEBUG_SEVERITY_HIGH            0x9146
#define GL_DEBUG_SEVERITY_MEDIUM          0x9147
#define GL_DEBUG_SEVERITY_LOW             0x9148
#define GL_DEBUG_SEVERITY_NOTIFICATION    0x826B
#define GL_DEBUG_SOURCE_API               0x8246
#define GL_DEBUG_SOURCE_WINDOW_SYSTEM     0x8247
#define GL_DEBUG_SOURCE_SHADER_COMPILER   0x8248
#define GL_DEBUG_SOURCE_THIRD_PARTY       0x8249
#define GL_DEBUG_SOURCE_APPLICATION       0x824A
#define GL_DEBUG_SOURCE_OTHER             0x824B
#define GL_DEBUG_TYPE_ERROR               0x824C
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR 0x824D
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR  0x824E
#define GL_DEBUG_TYPE_PORTABILITY         0x824F
#define GL_DEBUG_TYPE_PERFORMANCE         0x8250
#define GL_DEBUG_TYPE_OTHER               0x8251
#define GL_DEBUG_TYPE_MARKER              0x8268
#define GL_DEBUG_TYPE_PUSH_GROUP          0x8269
#define GL_DEBUG_TYPE_POP_GROUP           0x826A
#endif

}

#define _STR(x) #x
#define STR(x) _STR(x)
#define CONCAT2(s1, s2) s1 ## s2
#define CONCAT3(s1, s2, s3) s1 ## s2 ## s3
#define E(...) __VA_ARGS__


#ifndef SHADER_LIB_NAME
#define SHADER_LIB_NAME undefined
#endif

#define SHADER_LIB_NAME_STR STR(SHADER_LIB_NAME)
#define EXTERN_SHADER_SRC_VAR_H(name) CONCAT3(_binary_, name, _start)
#define EXTERN_SHADER_SRC_VAR EXTERN_SHADER_SRC_VAR_H(SHADER_LIB_NAME)

#define MAIN_SHADER_FUNCTION_H(name) void name ( glsl::uvec3 gl_GlobalInvocationID )
#define MAIN_SHADER_FUNCTION MAIN_SHADER_FUNCTION_H(SHADER_LIB_NAME)
#define MAIN_SHADER_FUNCTION_CALL SHADER_LIB_NAME

#define CPU_WRAPPER_FUNCTION_H(name) CONCAT2(name, _cpu_wrapper)
#define CPU_WRAPPER_FUNCTION CPU_WRAPPER_FUNCTION_H(SHADER_LIB_NAME)

#define CPU_INSTANCE_VAR_H(name) CONCAT2(name, _cpu_instance)
#define CPU_INSTANCE_VAR CPU_INSTANCE_VAR_H(SHADER_LIB_NAME)

#define GPU_OR_CPU(gpu, cpu) cpu

#define layout_init() using namespace glsl; extern shader_instance_cpu CPU_INSTANCE_VAR;
//#define layout(format, access, precision, type, name) auto name = type<format, access>(&CPU_INSTANCE_VAR, _STR(name))


template <int N, typename... Ts>
// recovers N-th typename from a parameter pack
using get_Nth_type = typename std::tuple_element_t<N, std::tuple<Ts...>>;

template <int N, typename... Ts>
// recovers N-th value of a corresponding type from a parameter pack
constexpr get_Nth_type<N, Ts...> get_Nth_value(Ts&&... values) { return std::get<N>(std::forward_as_tuple(std::forward<Ts>(values)...)); }


template<typename... Ts>
// concept is valid when all types in a parameter pack are the same
concept all_types_same = (std::same_as<get_Nth_type<0, Ts...>, Ts> and ...);

//static_assert(all_types_same<int, float, float>); // fail
//static_assert(all_types_same<float, float, float>); // OK


template<int... Is>
// concept is valid when all integers in a parameter pack are the same
concept all_ints_same = ((get_Nth_value<0>(Is...) == Is) and ...);

//static_assert(all_ints_same<1, 1, 2>); // fail
//static_assert(all_ints_same<1, 1, 1>); // OK


template <typename T, typename... Ts>
// concept is valid when a type T is among types in a parameter pack Ts
concept any_type_of = (std::same_as<T, Ts> or ...);

template <int I, int... Is>
// concept is valid when an integer I is among integers in a parameter pack Is
concept any_int_of = ((I == Is) or ...);


template<unsigned N, typename... Ts, unsigned... Seq>
// compile-time boolean is true when N-th type in a parameter pack is equal to any other type in the pack
constexpr bool any_type_same_as_Nth_helper(std::integer_sequence<unsigned, Seq...>) { return ((Seq != N && std::same_as<get_Nth_type<N, Ts...>, Ts>) or ...); }

template<unsigned N, typename... Ts>
// concept is valid when N-th type in a parameter pack is equal to any other type in the pack
concept any_type_same_as_Nth = (any_type_same_as_Nth_helper<N, Ts...>(std::make_integer_sequence<unsigned, sizeof...(Ts)>()));

//static_assert(any_type_same_as_Nth<0, int, float, float>); // fail
static_assert(any_type_same_as_Nth<1, int, float, float>); // OK

template<typename... Ts, unsigned... Seq>
// compile-time boolean is true is valid when any type in a parameter pack Ts contains a duplicate
constexpr bool any_type_same_helper(std::integer_sequence<unsigned, Seq...>) { return ((any_type_same_as_Nth<Seq, Ts...>) or ...); }

template<typename... Ts>
// concept is valid when all types in a parameter pack are unique
concept all_types_unique = (not any_type_same_helper<Ts...>(std::make_integer_sequence<unsigned, sizeof...(Ts)>()));

//static_assert(all_types_unique<int, float, int>); // fail
//static_assert(all_types_unique<float, float, int>); // fail
//static_assert(all_types_unique<int, float, double>); // OK


template<unsigned N, int... Is, unsigned... Seq>
// compile-time boolean is true when N-th integer in a parameter pack is equal to any other integer in the pack
constexpr bool any_int_same_as_Nth_helper(std::integer_sequence<unsigned, Seq...>) { return ((Seq != N && get_Nth_value<N>(Is...) == Is) or ...); }

template<unsigned N, int... Is>
// concept is valid when N-th integer in a parameter pack is equal to any other integer in the pack
concept any_int_same_as_Nth = (any_int_same_as_Nth_helper<N, Is...>(std::make_integer_sequence<unsigned, sizeof...(Is)>()));

//static_assert(any_int_same_as_Nth<0, 5, 4, 4>); // fail
//static_assert(any_int_same_as_Nth<2, 7, 7, 7, 7>); // OK

template<int... Is, unsigned... Seq>
// compile-time boolean is true is valid when any integer in a parameter pack Ts contains a duplicate
constexpr bool any_int_same_helper(std::integer_sequence<unsigned, Seq...>) { return ((any_int_same_as_Nth<Seq, Is...>) or ...); }

template<int... Is>
// concept is valid when all integers in a parameter pack are unique
concept all_ints_unique = (not any_int_same_helper<Is...>(std::make_integer_sequence<unsigned, sizeof...(Is)>()));

//static_assert(all_ints_unique<1, 1, 1>); // fail
//static_assert(all_ints_unique<5, 5, 1>); // fail
//static_assert(all_ints_unique<0, 1, 2>); // OK

template<class C_spec, template<auto...> class C_gen>
inline constexpr bool is_instance_of_v = std::false_type{};
template<template<auto...> class C_gen, auto... Params>
inline constexpr bool is_instance_of_v<C_gen<Params...>, C_gen> = std::true_type{};



namespace glsl {
    typedef unsigned int uint;

    template<class CHILD, unsigned int SIZE, typename SCALAR, typename SEQ>
    class vecX_common;

    template<unsigned int SIZE, typename SCALAR, unsigned int... I> requires (sizeof...(I) <= 4 and sizeof...(I) > 0 and SIZE <= 4 and SIZE > 0 and ((I < SIZE) and ...))
    class vecX_swizzle;

    template<typename SCALAR, unsigned int DIM> requires (DIM > 0)
    class vecX_data;

    template<unsigned int N, unsigned int M, typename SCALAR> // requires (N >= 2 and N <= 4 and M >= 2 and M <= 4)
    class matNxM;

    typedef enum GL_uniform_access {
        readwrite = 0,
        readonly = 1,
        writeonly = 2
    } uniform_access;

    typedef enum GL_memory_layout {
        shared = 0,
        packed = 1,
        std140 = 2,
        std430 = 3
    } mem_layout;

    class resource {
    public:
        std::string name;
        gles::GLuint64 type_enum;
        void* data;
    };

    template<typename VEC_SCALAR, unsigned int DIM, typename UNIT, uniform_access ACCESS>
    class imageXD;

    template<mem_layout MEM, typename UNIT>
    class buffer;

    class shader_instance_cpu {
    public:
        template<typename VEC_SCALAR, unsigned int DIM, typename UNIT, uniform_access ACCESS>
        void add_variable(imageXD<VEC_SCALAR, DIM, UNIT, ACCESS>* v_ptr, const char* name) {

        };

        template<mem_layout MEM, typename UNIT>
        void add_variable(buffer<MEM, UNIT>* v_ptr, const char* name) {

        };
    };

    template<typename T>
    inline constexpr unsigned int vecX_data_dim = 0UL;
    template<template<typename, unsigned int> typename T, typename SCALAR, unsigned int DIM>
    inline constexpr unsigned int vecX_data_dim<T<SCALAR, DIM>> = DIM;

    template <typename T, typename=void>
    struct vecX_converter;
    template <typename T>
    struct vecX_converter<T, typename std::enable_if<(std::is_arithmetic_v<T>)>::type> { using type = vecX_data<T, 1>; using scalar = T; };
    template <template<typename, unsigned int> typename T, typename SCALAR, unsigned int DIM>
    struct vecX_converter<T<SCALAR, DIM>, typename std::enable_if<(std::same_as<T<SCALAR, DIM>, vecX_data<SCALAR, DIM>>)>::type> { using type = vecX_data<SCALAR, DIM>; using scalar = SCALAR; };
    template <template<unsigned int, typename, unsigned int...> typename T, unsigned int SIZE, typename SCALAR, unsigned int... I>
    struct vecX_converter<T<SIZE, SCALAR, I...>, typename std::enable_if<(std::convertible_to<T<SIZE, SCALAR, I...>, vecX_data<SCALAR, sizeof...(I)>>)>::type> { using type = vecX_data<SCALAR, sizeof...(I)>; using scalar = SCALAR; };
    template <typename T>
    using vecX_compatible_t = typename vecX_converter<T>::type;
    template <typename T>
    concept vecX_compatible = not std::is_empty_v<vecX_compatible_t<T>>;
    template <typename T> requires (vecX_compatible<T>)
    inline constexpr unsigned int vecX_compatible_dim = vecX_data_dim<vecX_compatible_t<T>>;
    template <typename T> requires (vecX_compatible<T>)
    using vecX_compatible_scalar = typename vecX_converter<T>::scalar;

    template<typename SCALAR, unsigned int DIM1, unsigned int DIM2>
    auto operator ,(vecX_data<SCALAR, DIM1> lhs, vecX_data<SCALAR, DIM2> rhs) {
        vecX_data<SCALAR, DIM1+DIM2> ret(0);
        for (unsigned int i = 0; i < DIM1; i++) ret[i] = lhs[i];
        for (unsigned int j = 0; j < DIM2; j++) ret[DIM1+j] = rhs[j];
        return ret;
    }

    template<typename SCALAR, typename... Ts> requires ((vecX_compatible<Ts> and ...) and (std::convertible_to<vecX_compatible_scalar<Ts>, SCALAR> and ...))
    vecX_data<SCALAR, (vecX_compatible_dim<Ts> + ...)> vecX_concat(Ts... vs) {
        return ( vecX_data<SCALAR, vecX_compatible_dim<Ts>>(vs), ...);
    }

#define VECX_COMMON_AOP(__op__) \
    CHILD& operator __op__##=(const SCALAR rhs) requires (all_ints_unique<I...>) { \
        const unsigned int idx[sizeof...(I)] = { I... }; \
        auto lhs = static_cast<CHILD*>(this); \
        for (unsigned int i = 0; i < sizeof...(I); i++) lhs->data[idx[i]] __op__##= rhs; \
        return *lhs; \
    } \
    template<unsigned int SIZE2, unsigned int... J> requires (sizeof...(I) == sizeof...(J) and all_ints_unique<I...>) \
    CHILD& operator __op__##=(const vecX_swizzle<SIZE2, SCALAR, J...> rhs) { \
        const unsigned int idx[sizeof...(I)] = { I... }; \
        const unsigned int idx2[sizeof...(J)] = { J... }; \
        auto lhs = static_cast<CHILD*>(this); \
        for (unsigned int i = 0; i < sizeof...(I); i++) lhs->data[idx[i]] __op__##= rhs.data[idx2[i]]; \
        return *lhs; \
    } \
    CHILD& operator __op__##=(const vecX_data<SCALAR, sizeof...(I)> rhs) requires (all_ints_unique<I...>) { \
        auto lhs = static_cast<CHILD*>(this); \
        *lhs __op__##= rhs.self; \
        return *lhs; \
    } \
    template<typename T> requires (std::convertible_to<T, vecX_data<SCALAR, sizeof...(I)>> or std::convertible_to<T, SCALAR>) \
    vecX_data<SCALAR, sizeof...(I)> operator __op__(const T rhs) { \
        auto v3 = vecX_data<SCALAR, sizeof...(I)>(*static_cast<CHILD*>(this)); \
        v3.self __op__##= rhs; \
        return v3; \
    } \
    friend vecX_data<SCALAR, sizeof...(I)> operator __op__(SCALAR lhs, CHILD rhs) { \
        return vecX_data<SCALAR, sizeof...(I)>(lhs) __op__ rhs; \
    }

    template<class CHILD, unsigned int SIZE, typename SCALAR, unsigned int... I> requires (sizeof...(I) > 0 and SIZE > 0 and ((I < SIZE) and ...))
    class vecX_common<CHILD, SIZE, SCALAR, std::integer_sequence<unsigned int, I...>> {
    public:
        // overloading << for output stream
        friend std::ostream& operator<<(std::ostream& os, const CHILD& rhs) {
            const unsigned int idx[sizeof...(I)] = { I... };
            if constexpr (sizeof...(I) == 1) {
                os << rhs.data[idx[0]];
            } else {
                os << "(";
                for (unsigned int i = 0; i < sizeof...(I); i++) {
                    if (i > 0) os << ", ";
                    os << rhs.data[idx[i]];
                }
                os << ")";
            }
            return os;
        }

        // overload arithmetic & related assignment operators
        VECX_COMMON_AOP(+)
        VECX_COMMON_AOP(-)
        VECX_COMMON_AOP(*)
        VECX_COMMON_AOP(/)

        // overload subscript operator for array-like access to elements
        SCALAR& operator [](int i) {
            auto lhs = static_cast<CHILD*>(this);
            const unsigned int idx[sizeof...(I)] = { I... };
            return lhs->data[idx[i]];
        }

        // cast to a new vector of the proper dimension
        template<typename SCALAR2> requires (std::convertible_to<SCALAR, SCALAR2>)
        operator vecX_data<SCALAR2, sizeof...(I)>() const {
            return vecX_data<SCALAR2, sizeof...(I)>(*this);
        }

        // overloading cast operator for a different swizzle
        template<unsigned int... J>
        operator vecX_swizzle<SIZE, SCALAR, J...>() const {
            return *((vecX_swizzle<SIZE, SCALAR, J...>*) (this));
        }

        // helper to apply function in-place element-wise
        template<typename F, typename... T> requires ((std::convertible_to<T, vecX_data<SCALAR, sizeof...(I)>> or std::convertible_to<T, SCALAR>) and ...)
        auto apply(F func, T... other) {
            auto self = vecX_data<SCALAR, sizeof...(I)>(*static_cast<CHILD*>(this));
            for (unsigned int i = 0; i < sizeof...(I); i++) {
                self[i] = SCALAR(func(self[i], vecX_data<SCALAR, sizeof...(I)>(other)[i]...));
            }
            return self;
        }

        // returns L2 vector length
        SCALAR length() {
            SCALAR res = 0;
            vecX_data<SCALAR, sizeof...(I)> self(*static_cast<CHILD*>(this));
            for (unsigned int i = 0; i < sizeof...(I); i++) {
                res += self[i] * self[i];
            }
            return SCALAR(::sqrt(res));
        }

        template<typename T> requires (std::convertible_to<T, vecX_data<SCALAR, sizeof...(I)>>)
        SCALAR dot(T vec) {
            SCALAR res = 0;
            vecX_data<SCALAR, sizeof...(I)> self(*static_cast<CHILD*>(this));
            vecX_data<SCALAR, sizeof...(I)> another(vec);
            for (unsigned int i = 0; i < sizeof...(I); i++) {
                res += self[i] * another[i];
            }
            return res;
        }
    };

    template<unsigned int SIZE, typename SCALAR, unsigned int... I> requires (sizeof...(I) <= 4 and sizeof...(I) > 0 and SIZE <= 4 and SIZE > 0 and ((I < SIZE) and ...))
    class vecX_swizzle : public vecX_common<vecX_swizzle<SIZE, SCALAR, I...>, SIZE, SCALAR, std::integer_sequence<unsigned int, I...>> {
    public:
        SCALAR data[SIZE];

        // single-element vector swizzle cast to scalar
        operator SCALAR() const requires (sizeof...(I) == 1) {
            return data[get_Nth_value<0>(I...)];
        }

        // assignment of scalar value to all (unique) elements of the swizzle
        vecX_swizzle& operator=(const SCALAR x) requires (all_ints_unique<I...>) {
            const unsigned int idx[sizeof...(I)] = { I... };
            for (unsigned int i = 0; i < sizeof...(I); i++) data[idx[i]] = x;
            return *this;
        }

        // assignment of compatible vector to unique elements of the swizzle
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR> and all_ints_unique<I...>)
        vecX_swizzle& operator=(const vecX_data<SCALAR2, sizeof...(I)> rhs) {
            *this = rhs.self;
            return *this;
        }

        // assignment of compatible vector swizzle to unique elements of the swizzle
        template<unsigned int SIZE2, typename SCALAR2, unsigned int... J> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == sizeof...(J) and all_ints_unique<I...>)
        vecX_swizzle& operator=(const vecX_swizzle<SIZE2, SCALAR2, J...> rhs) {
            const unsigned int idx[sizeof...(I)] = { I... };
            const unsigned int idx2[sizeof...(J)] = { J... };
            for (unsigned int i = 0; i < sizeof...(I); i++) data[idx[i]] = SCALAR(rhs.data[idx2[i]]);
            return *this;
        }
    };


    template<typename SCALAR, unsigned int DIM> requires (DIM > 0)
    class vecX_data : public vecX_common<vecX_data<SCALAR, DIM>, DIM, SCALAR, std::make_integer_sequence<unsigned int, DIM>> {
    public:
        SCALAR data[DIM];

        vecX_data(SCALAR all) { for (unsigned int i = 0; i < DIM; i++) data[i] = all; };
    };

    // 1-dim vectors:
    // - combinations: 1**1+1**2+1**3+1**4 = 1+1+1+1 = 4
    template<typename SCALAR>
    class vecX_data<SCALAR, 1> : public vecX_common<vecX_data<SCALAR, 1>, 1, SCALAR, std::make_integer_sequence<unsigned int, 1>> {
    public:
        union {
            SCALAR data[1];
            vecX_swizzle<1, SCALAR, 0> x, r, s, self;
            vecX_swizzle<1, SCALAR, 0, 0> xx, rr, ss;
            vecX_swizzle<1, SCALAR, 0, 0, 0> xxx, rrr, sss;
            vecX_swizzle<1, SCALAR, 0, 0, 0, 0> xxxx, rrrr, ssss;
        };

        vecX_data(SCALAR x) { self = x; }

        vecX_data(vecX_data& x) { self = SCALAR(x); }

        template<typename SCALAR2, unsigned int DIM> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(vecX_data<SCALAR2, DIM> v) { self = v.x; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> v) { self = static_cast<vecX_swizzle<SIZE, SCALAR2, get_Nth_value<0>(I...)>>(v); }

        vecX_data& operator=(SCALAR v) {
            self = v;
            return *this;
        }

        operator SCALAR() const {
            return data[0];
        }
    };

    // 2-dim vectors:
    // - combinations: 2**1+2**2+2**3+2**4 = 2+4+8+16 = 30
    template<typename SCALAR>
    class vecX_data<SCALAR, 2> : public vecX_common<vecX_data<SCALAR, 2>, 2, SCALAR, std::make_integer_sequence<unsigned int, 2>> {
    public:
        union {
            SCALAR data[2];
            vecX_swizzle<2, SCALAR, 0> x, r, s;
            vecX_swizzle<2, SCALAR, 1> y, g, t;
            vecX_swizzle<2, SCALAR, 0, 0> xx, rr, ss;
            vecX_swizzle<2, SCALAR, 0, 1> xy, rg, st, self;
            vecX_swizzle<2, SCALAR, 1, 0> yx, gr, ts;
            vecX_swizzle<2, SCALAR, 1, 1> yy, gg, tt;
            vecX_swizzle<2, SCALAR, 0, 0, 0> xxx, rrr, sss;
            vecX_swizzle<2, SCALAR, 0, 0, 1> xxy, rrg, sst;
            vecX_swizzle<2, SCALAR, 0, 1, 0> xyx, rgr, sts;
            vecX_swizzle<2, SCALAR, 0, 1, 1> xyy, rgg, stt;
            vecX_swizzle<2, SCALAR, 1, 0, 0> yxx, grr, tss;
            vecX_swizzle<2, SCALAR, 1, 0, 1> yxy, grg, tst;
            vecX_swizzle<2, SCALAR, 1, 1, 0> yyx, ggr, tts;
            vecX_swizzle<2, SCALAR, 1, 1, 1> yyy, ggg, ttt;
            vecX_swizzle<2, SCALAR, 0, 0, 0, 0> xxxx, rrrr, ssss;
            vecX_swizzle<2, SCALAR, 0, 0, 0, 1> xxxy, rrrg, ssst;
            vecX_swizzle<2, SCALAR, 0, 0, 1, 0> xxyx, rrgr, ssts;
            vecX_swizzle<2, SCALAR, 0, 0, 1, 1> xxyy, rrgg, sstt;
            vecX_swizzle<2, SCALAR, 0, 1, 0, 0> xyxx, rgrr, stss;
            vecX_swizzle<2, SCALAR, 0, 1, 0, 1> xyxy, rgrg, stst;
            vecX_swizzle<2, SCALAR, 0, 1, 1, 0> xyyx, rggr, stts;
            vecX_swizzle<2, SCALAR, 0, 1, 1, 1> xyyy, rggg, sttt;
            vecX_swizzle<2, SCALAR, 1, 0, 0, 0> yxxx, grrr, tsss;
            vecX_swizzle<2, SCALAR, 1, 0, 0, 1> yxxy, grrg, tsst;
            vecX_swizzle<2, SCALAR, 1, 0, 1, 0> yxyx, grgr, tsts;
            vecX_swizzle<2, SCALAR, 1, 0, 1, 1> yxyy, grgg, tstt;
            vecX_swizzle<2, SCALAR, 1, 1, 0, 0> yyxx, ggrr, ttss;
            vecX_swizzle<2, SCALAR, 1, 1, 0, 1> yyxy, ggrg, ttst;
            vecX_swizzle<2, SCALAR, 1, 1, 1, 0> yyyx, gggr, ttts;
            vecX_swizzle<2, SCALAR, 1, 1, 1, 1> yyyy, gggg, tttt;
        };

        explicit vecX_data(SCALAR all) { self = all; }

        explicit vecX_data(SCALAR x, SCALAR y) { r = x; g = y; }

        template<typename SCALAR2, unsigned int DIM> requires (std::convertible_to<SCALAR2, SCALAR> and DIM >= 2)
        explicit vecX_data(vecX_data<SCALAR2, DIM> v) { self = v.xy; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) >= 2)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> v) { self = static_cast<vecX_swizzle<SIZE, SCALAR2, get_Nth_value<0>(I...), get_Nth_value<1>(I...)>>(v); }

        vecX_data& operator=(SCALAR v) {
            self = v;
            return *this;
        }

        template<unsigned int SIZE, unsigned int... I> requires (sizeof...(I) == 2)
        vecX_data& operator=(vecX_swizzle<SIZE, SCALAR, I...> v) {
            self = v;
            return *this;
        }
    };

    // 3-dim vectors:
    // - combinations: 3**1+3**2+3**3+3**4 = 3+9+27+81 = 120
    template<typename SCALAR>
    class vecX_data<SCALAR, 3> : public vecX_common<vecX_data<SCALAR, 3>, 3, SCALAR, std::make_integer_sequence<unsigned int, 3>> {
    public:
        union {
            SCALAR data[3];
            vecX_swizzle<3, SCALAR, 0> x, r, s;
            vecX_swizzle<3, SCALAR, 1> y, g, t;
            vecX_swizzle<3, SCALAR, 2> z, b, p;
            vecX_swizzle<3, SCALAR, 0, 0> xx, rr, ss;
            vecX_swizzle<3, SCALAR, 0, 1> xy, rg, st;
            vecX_swizzle<3, SCALAR, 0, 2> xz, rb, sp;
            vecX_swizzle<3, SCALAR, 1, 0> yx, gr, ts;
            vecX_swizzle<3, SCALAR, 1, 1> yy, gg, tt;
            vecX_swizzle<3, SCALAR, 1, 2> yz, gb, tp;
            vecX_swizzle<3, SCALAR, 2, 0> zx, br, ps;
            vecX_swizzle<3, SCALAR, 2, 1> zy, bg, pt;
            vecX_swizzle<3, SCALAR, 2, 2> zz, bb, pp;
            vecX_swizzle<3, SCALAR, 0, 0, 0> xxx, rrr, sss;
            vecX_swizzle<3, SCALAR, 0, 0, 1> xxy, rrg, sst;
            vecX_swizzle<3, SCALAR, 0, 0, 2> xxz, rrb, ssp;
            vecX_swizzle<3, SCALAR, 0, 1, 0> xyx, rgr, sts;
            vecX_swizzle<3, SCALAR, 0, 1, 1> xyy, rgg, stt;
            vecX_swizzle<3, SCALAR, 0, 1, 2> xyz, rgb, stp, self;
            vecX_swizzle<3, SCALAR, 0, 2, 0> xzx, rbr, sps;
            vecX_swizzle<3, SCALAR, 0, 2, 1> xzy, rbg, spt;
            vecX_swizzle<3, SCALAR, 0, 2, 2> xzz, rbb, spp;
            vecX_swizzle<3, SCALAR, 1, 0, 0> yxx, grr, tss;
            vecX_swizzle<3, SCALAR, 1, 0, 1> yxy, grg, tst;
            vecX_swizzle<3, SCALAR, 1, 0, 2> yxz, grb, tsp;
            vecX_swizzle<3, SCALAR, 1, 1, 0> yyx, ggr, tts;
            vecX_swizzle<3, SCALAR, 1, 1, 1> yyy, ggg, ttt;
            vecX_swizzle<3, SCALAR, 1, 1, 2> yyz, ggb, ttp;
            vecX_swizzle<3, SCALAR, 1, 2, 0> yzx, gbr, tps;
            vecX_swizzle<3, SCALAR, 1, 2, 1> yzy, gbg, tpt;
            vecX_swizzle<3, SCALAR, 1, 2, 2> yzz, gbb, tpp;
            vecX_swizzle<3, SCALAR, 2, 0, 0> zxx, brr, pss;
            vecX_swizzle<3, SCALAR, 2, 0, 1> zxy, brg, pst;
            vecX_swizzle<3, SCALAR, 2, 0, 2> zxz, brb, psp;
            vecX_swizzle<3, SCALAR, 2, 1, 0> zyx, bgr, pts;
            vecX_swizzle<3, SCALAR, 2, 1, 1> zyy, bgg, ptt;
            vecX_swizzle<3, SCALAR, 2, 1, 2> zyz, bgb, ptp;
            vecX_swizzle<3, SCALAR, 2, 2, 0> zzx, bbr, pps;
            vecX_swizzle<3, SCALAR, 2, 2, 1> zzy, bbg, ppt;
            vecX_swizzle<3, SCALAR, 2, 2, 2> zzz, bbb, ppp;
            vecX_swizzle<3, SCALAR, 0, 0, 0, 0> xxxx, rrrr, ssss;
            vecX_swizzle<3, SCALAR, 0, 0, 0, 1> xxxy, rrrg, ssst;
            vecX_swizzle<3, SCALAR, 0, 0, 0, 2> xxxz, rrrb, sssp;
            vecX_swizzle<3, SCALAR, 0, 0, 1, 0> xxyx, rrgr, ssts;
            vecX_swizzle<3, SCALAR, 0, 0, 1, 1> xxyy, rrgg, sstt;
            vecX_swizzle<3, SCALAR, 0, 0, 1, 2> xxyz, rrgb, sstp;
            vecX_swizzle<3, SCALAR, 0, 0, 2, 0> xxzx, rrbr, ssps;
            vecX_swizzle<3, SCALAR, 0, 0, 2, 1> xxzy, rrbg, sspt;
            vecX_swizzle<3, SCALAR, 0, 0, 2, 2> xxzz, rrbb, sspp;
            vecX_swizzle<3, SCALAR, 0, 1, 0, 0> xyxx, rgrr, stss;
            vecX_swizzle<3, SCALAR, 0, 1, 0, 1> xyxy, rgrg, stst;
            vecX_swizzle<3, SCALAR, 0, 1, 0, 2> xyxz, rgrb, stsp;
            vecX_swizzle<3, SCALAR, 0, 1, 1, 0> xyyx, rggr, stts;
            vecX_swizzle<3, SCALAR, 0, 1, 1, 1> xyyy, rggg, sttt;
            vecX_swizzle<3, SCALAR, 0, 1, 1, 2> xyyz, rggb, sttp;
            vecX_swizzle<3, SCALAR, 0, 1, 2, 0> xyzx, rgbr, stps;
            vecX_swizzle<3, SCALAR, 0, 1, 2, 1> xyzy, rgbg, stpt;
            vecX_swizzle<3, SCALAR, 0, 1, 2, 2> xyzz, rgbb, stpp;
            vecX_swizzle<3, SCALAR, 0, 2, 0, 0> xzxx, rbrr, spss;
            vecX_swizzle<3, SCALAR, 0, 2, 0, 1> xzxy, rbrg, spst;
            vecX_swizzle<3, SCALAR, 0, 2, 0, 2> xzxz, rbrb, spsp;
            vecX_swizzle<3, SCALAR, 0, 2, 1, 0> xzyx, rbgr, spts;
            vecX_swizzle<3, SCALAR, 0, 2, 1, 1> xzyy, rbgg, sptt;
            vecX_swizzle<3, SCALAR, 0, 2, 1, 2> xzyz, rbgb, sptp;
            vecX_swizzle<3, SCALAR, 0, 2, 2, 0> xzzx, rbbr, spps;
            vecX_swizzle<3, SCALAR, 0, 2, 2, 1> xzzy, rbbg, sppt;
            vecX_swizzle<3, SCALAR, 0, 2, 2, 2> xzzz, rbbb, sppp;
            vecX_swizzle<3, SCALAR, 1, 0, 0, 0> yxxx, grrr, tsss;
            vecX_swizzle<3, SCALAR, 1, 0, 0, 1> yxxy, grrg, tsst;
            vecX_swizzle<3, SCALAR, 1, 0, 0, 2> yxxz, grrb, tssp;
            vecX_swizzle<3, SCALAR, 1, 0, 1, 0> yxyx, grgr, tsts;
            vecX_swizzle<3, SCALAR, 1, 0, 1, 1> yxyy, grgg, tstt;
            vecX_swizzle<3, SCALAR, 1, 0, 1, 2> yxyz, grgb, tstp;
            vecX_swizzle<3, SCALAR, 1, 0, 2, 0> yxzx, grbr, tsps;
            vecX_swizzle<3, SCALAR, 1, 0, 2, 1> yxzy, grbg, tspt;
            vecX_swizzle<3, SCALAR, 1, 0, 2, 2> yxzz, grbb, tspp;
            vecX_swizzle<3, SCALAR, 1, 1, 0, 0> yyxx, ggrr, ttss;
            vecX_swizzle<3, SCALAR, 1, 1, 0, 1> yyxy, ggrg, ttst;
            vecX_swizzle<3, SCALAR, 1, 1, 0, 2> yyxz, ggrb, ttsp;
            vecX_swizzle<3, SCALAR, 1, 1, 1, 0> yyyx, gggr, ttts;
            vecX_swizzle<3, SCALAR, 1, 1, 1, 1> yyyy, gggg, tttt;
            vecX_swizzle<3, SCALAR, 1, 1, 1, 2> yyyz, gggb, tttp;
            vecX_swizzle<3, SCALAR, 1, 1, 2, 0> yyzx, ggbr, ttps;
            vecX_swizzle<3, SCALAR, 1, 1, 2, 1> yyzy, ggbg, ttpt;
            vecX_swizzle<3, SCALAR, 1, 1, 2, 2> yyzz, ggbb, ttpp;
            vecX_swizzle<3, SCALAR, 1, 2, 0, 0> yzxx, gbrr, tpss;
            vecX_swizzle<3, SCALAR, 1, 2, 0, 1> yzxy, gbrg, tpst;
            vecX_swizzle<3, SCALAR, 1, 2, 0, 2> yzxz, gbrb, tpsp;
            vecX_swizzle<3, SCALAR, 1, 2, 1, 0> yzyx, gbgr, tpts;
            vecX_swizzle<3, SCALAR, 1, 2, 1, 1> yzyy, gbgg, tptt;
            vecX_swizzle<3, SCALAR, 1, 2, 1, 2> yzyz, gbgb, tptp;
            vecX_swizzle<3, SCALAR, 1, 2, 2, 0> yzzx, gbbr, tpps;
            vecX_swizzle<3, SCALAR, 1, 2, 2, 1> yzzy, gbbg, tppt;
            vecX_swizzle<3, SCALAR, 1, 2, 2, 2> yzzz, gbbb, tppp;
            vecX_swizzle<3, SCALAR, 2, 0, 0, 0> zxxx, brrr, psss;
            vecX_swizzle<3, SCALAR, 2, 0, 0, 1> zxxy, brrg, psst;
            vecX_swizzle<3, SCALAR, 2, 0, 0, 2> zxxz, brrb, pssp;
            vecX_swizzle<3, SCALAR, 2, 0, 1, 0> zxyx, brgr, psts;
            vecX_swizzle<3, SCALAR, 2, 0, 1, 1> zxyy, brgg, pstt;
            vecX_swizzle<3, SCALAR, 2, 0, 1, 2> zxyz, brgb, pstp;
            vecX_swizzle<3, SCALAR, 2, 0, 2, 0> zxzx, brbr, psps;
            vecX_swizzle<3, SCALAR, 2, 0, 2, 1> zxzy, brbg, pspt;
            vecX_swizzle<3, SCALAR, 2, 0, 2, 2> zxzz, brbb, pspp;
            vecX_swizzle<3, SCALAR, 2, 1, 0, 0> zyxx, bgrr, ptss;
            vecX_swizzle<3, SCALAR, 2, 1, 0, 1> zyxy, bgrg, ptst;
            vecX_swizzle<3, SCALAR, 2, 1, 0, 2> zyxz, bgrb, ptsp;
            vecX_swizzle<3, SCALAR, 2, 1, 1, 0> zyyx, bggr, ptts;
            vecX_swizzle<3, SCALAR, 2, 1, 1, 1> zyyy, bggg, pttt;
            vecX_swizzle<3, SCALAR, 2, 1, 1, 2> zyyz, bggb, pttp;
            vecX_swizzle<3, SCALAR, 2, 1, 2, 0> zyzx, bgbr, ptps;
            vecX_swizzle<3, SCALAR, 2, 1, 2, 1> zyzy, bgbg, ptpt;
            vecX_swizzle<3, SCALAR, 2, 1, 2, 2> zyzz, bgbb, ptpp;
            vecX_swizzle<3, SCALAR, 2, 2, 0, 0> zzxx, bbrr, ppss;
            vecX_swizzle<3, SCALAR, 2, 2, 0, 1> zzxy, bbrg, ppst;
            vecX_swizzle<3, SCALAR, 2, 2, 0, 2> zzxz, bbrb, ppsp;
            vecX_swizzle<3, SCALAR, 2, 2, 1, 0> zzyx, bbgr, ppts;
            vecX_swizzle<3, SCALAR, 2, 2, 1, 1> zzyy, bbgg, pptt;
            vecX_swizzle<3, SCALAR, 2, 2, 1, 2> zzyz, bbgb, pptp;
            vecX_swizzle<3, SCALAR, 2, 2, 2, 0> zzzx, bbbr, ppps;
            vecX_swizzle<3, SCALAR, 2, 2, 2, 1> zzzy, bbbg, pppt;
            vecX_swizzle<3, SCALAR, 2, 2, 2, 2> zzzz, bbbb, pppp;
        };

        explicit vecX_data(SCALAR all) { self = all; }

        explicit vecX_data(SCALAR x, SCALAR y, SCALAR z) { r = x; g = y; b = z; }

        template<typename SCALAR2, unsigned int DIM> requires (std::convertible_to<SCALAR2, SCALAR> and DIM >= 3)
        explicit vecX_data(vecX_data<SCALAR2, DIM> v) { self = v.xyz; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) >= 3)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> v) { self = static_cast<vecX_swizzle<SIZE, SCALAR2, get_Nth_value<0>(I...), get_Nth_value<1>(I...), get_Nth_value<2>(I...)>>(v); }


        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(vecX_data<SCALAR2, 2> xy, SCALAR z) { rg = xy; b = z; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 2)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> xy, SCALAR z) { rg = xy; b = z; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(SCALAR x, vecX_data<SCALAR2, 2> yz) { r = x; gb = yz; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 2)
        explicit vecX_data(SCALAR x, vecX_swizzle<SIZE, SCALAR2, I...> yz) { r = x; gb = yz; }


        vecX_data& operator=(SCALAR v) {
            self = v;
            return *this;
        }

        template<unsigned int SIZE, unsigned int... I> requires (sizeof...(I) == 3)
        vecX_data& operator=(vecX_swizzle<SIZE, SCALAR, I...> v) {
            self = v;
            return *this;
        }
    };


    // 4-dim vectors:
    // - combinations: 4**1+4**2+4**3+4**4 = 4+16+64+256 = 340
    template<typename SCALAR>
    class vecX_data<SCALAR, 4> : public vecX_common<vecX_data<SCALAR, 4>, 4, SCALAR, std::make_integer_sequence<unsigned int, 4>> {
    public:
        union {
            SCALAR data[4];
            vecX_swizzle<4, SCALAR, 0> x, r, s;
            vecX_swizzle<4, SCALAR, 1> y, g, t;
            vecX_swizzle<4, SCALAR, 2> z, b, p;
            vecX_swizzle<4, SCALAR, 3> w, a, q;
            vecX_swizzle<4, SCALAR, 0, 0> xx, rr, ss;
            vecX_swizzle<4, SCALAR, 0, 1> xy, rg, st;
            vecX_swizzle<4, SCALAR, 0, 2> xz, rb, sp;
            vecX_swizzle<4, SCALAR, 0, 3> xw, ra, sq;
            vecX_swizzle<4, SCALAR, 1, 0> yx, gr, ts;
            vecX_swizzle<4, SCALAR, 1, 1> yy, gg, tt;
            vecX_swizzle<4, SCALAR, 1, 2> yz, gb, tp;
            vecX_swizzle<4, SCALAR, 1, 3> yw, ga, tq;
            vecX_swizzle<4, SCALAR, 2, 0> zx, br, ps;
            vecX_swizzle<4, SCALAR, 2, 1> zy, bg, pt;
            vecX_swizzle<4, SCALAR, 2, 2> zz, bb, pp;
            vecX_swizzle<4, SCALAR, 2, 3> zw, ba, pq;
            vecX_swizzle<4, SCALAR, 3, 0> wx, ar, qs;
            vecX_swizzle<4, SCALAR, 3, 1> wy, ag, qt;
            vecX_swizzle<4, SCALAR, 3, 2> wz, ab, qp;
            vecX_swizzle<4, SCALAR, 3, 3> ww, aa, qq;
            vecX_swizzle<4, SCALAR, 0, 0, 0> xxx, rrr, sss;
            vecX_swizzle<4, SCALAR, 0, 0, 1> xxy, rrg, sst;
            vecX_swizzle<4, SCALAR, 0, 0, 2> xxz, rrb, ssp;
            vecX_swizzle<4, SCALAR, 0, 0, 3> xxw, rra, ssq;
            vecX_swizzle<4, SCALAR, 0, 1, 0> xyx, rgr, sts;
            vecX_swizzle<4, SCALAR, 0, 1, 1> xyy, rgg, stt;
            vecX_swizzle<4, SCALAR, 0, 1, 2> xyz, rgb, stp;
            vecX_swizzle<4, SCALAR, 0, 1, 3> xyw, rga, stq;
            vecX_swizzle<4, SCALAR, 0, 2, 0> xzx, rbr, sps;
            vecX_swizzle<4, SCALAR, 0, 2, 1> xzy, rbg, spt;
            vecX_swizzle<4, SCALAR, 0, 2, 2> xzz, rbb, spp;
            vecX_swizzle<4, SCALAR, 0, 2, 3> xzw, rba, spq;
            vecX_swizzle<4, SCALAR, 0, 3, 0> xwx, rar, sqs;
            vecX_swizzle<4, SCALAR, 0, 3, 1> xwy, rag, sqt;
            vecX_swizzle<4, SCALAR, 0, 3, 2> xwz, rab, sqp;
            vecX_swizzle<4, SCALAR, 0, 3, 3> xww, raa, sqq;
            vecX_swizzle<4, SCALAR, 1, 0, 0> yxx, grr, tss;
            vecX_swizzle<4, SCALAR, 1, 0, 1> yxy, grg, tst;
            vecX_swizzle<4, SCALAR, 1, 0, 2> yxz, grb, tsp;
            vecX_swizzle<4, SCALAR, 1, 0, 3> yxw, gra, tsq;
            vecX_swizzle<4, SCALAR, 1, 1, 0> yyx, ggr, tts;
            vecX_swizzle<4, SCALAR, 1, 1, 1> yyy, ggg, ttt;
            vecX_swizzle<4, SCALAR, 1, 1, 2> yyz, ggb, ttp;
            vecX_swizzle<4, SCALAR, 1, 1, 3> yyw, gga, ttq;
            vecX_swizzle<4, SCALAR, 1, 2, 0> yzx, gbr, tps;
            vecX_swizzle<4, SCALAR, 1, 2, 1> yzy, gbg, tpt;
            vecX_swizzle<4, SCALAR, 1, 2, 2> yzz, gbb, tpp;
            vecX_swizzle<4, SCALAR, 1, 2, 3> yzw, gba, tpq;
            vecX_swizzle<4, SCALAR, 1, 3, 0> ywx, gar, tqs;
            vecX_swizzle<4, SCALAR, 1, 3, 1> ywy, gag, tqt;
            vecX_swizzle<4, SCALAR, 1, 3, 2> ywz, gab, tqp;
            vecX_swizzle<4, SCALAR, 1, 3, 3> yww, gaa, tqq;
            vecX_swizzle<4, SCALAR, 2, 0, 0> zxx, brr, pss;
            vecX_swizzle<4, SCALAR, 2, 0, 1> zxy, brg, pst;
            vecX_swizzle<4, SCALAR, 2, 0, 2> zxz, brb, psp;
            vecX_swizzle<4, SCALAR, 2, 0, 3> zxw, bra, psq;
            vecX_swizzle<4, SCALAR, 2, 1, 0> zyx, bgr, pts;
            vecX_swizzle<4, SCALAR, 2, 1, 1> zyy, bgg, ptt;
            vecX_swizzle<4, SCALAR, 2, 1, 2> zyz, bgb, ptp;
            vecX_swizzle<4, SCALAR, 2, 1, 3> zyw, bga, ptq;
            vecX_swizzle<4, SCALAR, 2, 2, 0> zzx, bbr, pps;
            vecX_swizzle<4, SCALAR, 2, 2, 1> zzy, bbg, ppt;
            vecX_swizzle<4, SCALAR, 2, 2, 2> zzz, bbb, ppp;
            vecX_swizzle<4, SCALAR, 2, 2, 3> zzw, bba, ppq;
            vecX_swizzle<4, SCALAR, 2, 3, 0> zwx, bar, pqs;
            vecX_swizzle<4, SCALAR, 2, 3, 1> zwy, bag, pqt;
            vecX_swizzle<4, SCALAR, 2, 3, 2> zwz, bab, pqp;
            vecX_swizzle<4, SCALAR, 2, 3, 3> zww, baa, pqq;
            vecX_swizzle<4, SCALAR, 3, 0, 0> wxx, arr, qss;
            vecX_swizzle<4, SCALAR, 3, 0, 1> wxy, arg, qst;
            vecX_swizzle<4, SCALAR, 3, 0, 2> wxz, arb, qsp;
            vecX_swizzle<4, SCALAR, 3, 0, 3> wxw, ara, qsq;
            vecX_swizzle<4, SCALAR, 3, 1, 0> wyx, agr, qts;
            vecX_swizzle<4, SCALAR, 3, 1, 1> wyy, agg, qtt;
            vecX_swizzle<4, SCALAR, 3, 1, 2> wyz, agb, qtp;
            vecX_swizzle<4, SCALAR, 3, 1, 3> wyw, aga, qtq;
            vecX_swizzle<4, SCALAR, 3, 2, 0> wzx, abr, qps;
            vecX_swizzle<4, SCALAR, 3, 2, 1> wzy, abg, qpt;
            vecX_swizzle<4, SCALAR, 3, 2, 2> wzz, abb, qpp;
            vecX_swizzle<4, SCALAR, 3, 2, 3> wzw, aba, qpq;
            vecX_swizzle<4, SCALAR, 3, 3, 0> wwx, aar, qqs;
            vecX_swizzle<4, SCALAR, 3, 3, 1> wwy, aag, qqt;
            vecX_swizzle<4, SCALAR, 3, 3, 2> wwz, aab, qqp;
            vecX_swizzle<4, SCALAR, 3, 3, 3> www, aaa, qqq;
            vecX_swizzle<4, SCALAR, 0, 0, 0, 0> xxxx, rrrr, ssss;
            vecX_swizzle<4, SCALAR, 0, 0, 0, 1> xxxy, rrrg, ssst;
            vecX_swizzle<4, SCALAR, 0, 0, 0, 2> xxxz, rrrb, sssp;
            vecX_swizzle<4, SCALAR, 0, 0, 0, 3> xxxw, rrra, sssq;
            vecX_swizzle<4, SCALAR, 0, 0, 1, 0> xxyx, rrgr, ssts;
            vecX_swizzle<4, SCALAR, 0, 0, 1, 1> xxyy, rrgg, sstt;
            vecX_swizzle<4, SCALAR, 0, 0, 1, 2> xxyz, rrgb, sstp;
            vecX_swizzle<4, SCALAR, 0, 0, 1, 3> xxyw, rrga, sstq;
            vecX_swizzle<4, SCALAR, 0, 0, 2, 0> xxzx, rrbr, ssps;
            vecX_swizzle<4, SCALAR, 0, 0, 2, 1> xxzy, rrbg, sspt;
            vecX_swizzle<4, SCALAR, 0, 0, 2, 2> xxzz, rrbb, sspp;
            vecX_swizzle<4, SCALAR, 0, 0, 2, 3> xxzw, rrba, sspq;
            vecX_swizzle<4, SCALAR, 0, 0, 3, 0> xxwx, rrar, ssqs;
            vecX_swizzle<4, SCALAR, 0, 0, 3, 1> xxwy, rrag, ssqt;
            vecX_swizzle<4, SCALAR, 0, 0, 3, 2> xxwz, rrab, ssqp;
            vecX_swizzle<4, SCALAR, 0, 0, 3, 3> xxww, rraa, ssqq;
            vecX_swizzle<4, SCALAR, 0, 1, 0, 0> xyxx, rgrr, stss;
            vecX_swizzle<4, SCALAR, 0, 1, 0, 1> xyxy, rgrg, stst;
            vecX_swizzle<4, SCALAR, 0, 1, 0, 2> xyxz, rgrb, stsp;
            vecX_swizzle<4, SCALAR, 0, 1, 0, 3> xyxw, rgra, stsq;
            vecX_swizzle<4, SCALAR, 0, 1, 1, 0> xyyx, rggr, stts;
            vecX_swizzle<4, SCALAR, 0, 1, 1, 1> xyyy, rggg, sttt;
            vecX_swizzle<4, SCALAR, 0, 1, 1, 2> xyyz, rggb, sttp;
            vecX_swizzle<4, SCALAR, 0, 1, 1, 3> xyyw, rgga, sttq;
            vecX_swizzle<4, SCALAR, 0, 1, 2, 0> xyzx, rgbr, stps;
            vecX_swizzle<4, SCALAR, 0, 1, 2, 1> xyzy, rgbg, stpt;
            vecX_swizzle<4, SCALAR, 0, 1, 2, 2> xyzz, rgbb, stpp;
            vecX_swizzle<4, SCALAR, 0, 1, 2, 3> xyzw, rgba, stpq, self;
            vecX_swizzle<4, SCALAR, 0, 1, 3, 0> xywx, rgar, stqs;
            vecX_swizzle<4, SCALAR, 0, 1, 3, 1> xywy, rgag, stqt;
            vecX_swizzle<4, SCALAR, 0, 1, 3, 2> xywz, rgab, stqp;
            vecX_swizzle<4, SCALAR, 0, 1, 3, 3> xyww, rgaa, stqq;
            vecX_swizzle<4, SCALAR, 0, 2, 0, 0> xzxx, rbrr, spss;
            vecX_swizzle<4, SCALAR, 0, 2, 0, 1> xzxy, rbrg, spst;
            vecX_swizzle<4, SCALAR, 0, 2, 0, 2> xzxz, rbrb, spsp;
            vecX_swizzle<4, SCALAR, 0, 2, 0, 3> xzxw, rbra, spsq;
            vecX_swizzle<4, SCALAR, 0, 2, 1, 0> xzyx, rbgr, spts;
            vecX_swizzle<4, SCALAR, 0, 2, 1, 1> xzyy, rbgg, sptt;
            vecX_swizzle<4, SCALAR, 0, 2, 1, 2> xzyz, rbgb, sptp;
            vecX_swizzle<4, SCALAR, 0, 2, 1, 3> xzyw, rbga, sptq;
            vecX_swizzle<4, SCALAR, 0, 2, 2, 0> xzzx, rbbr, spps;
            vecX_swizzle<4, SCALAR, 0, 2, 2, 1> xzzy, rbbg, sppt;
            vecX_swizzle<4, SCALAR, 0, 2, 2, 2> xzzz, rbbb, sppp;
            vecX_swizzle<4, SCALAR, 0, 2, 2, 3> xzzw, rbba, sppq;
            vecX_swizzle<4, SCALAR, 0, 2, 3, 0> xzwx, rbar, spqs;
            vecX_swizzle<4, SCALAR, 0, 2, 3, 1> xzwy, rbag, spqt;
            vecX_swizzle<4, SCALAR, 0, 2, 3, 2> xzwz, rbab, spqp;
            vecX_swizzle<4, SCALAR, 0, 2, 3, 3> xzww, rbaa, spqq;
            vecX_swizzle<4, SCALAR, 0, 3, 0, 0> xwxx, rarr, sqss;
            vecX_swizzle<4, SCALAR, 0, 3, 0, 1> xwxy, rarg, sqst;
            vecX_swizzle<4, SCALAR, 0, 3, 0, 2> xwxz, rarb, sqsp;
            vecX_swizzle<4, SCALAR, 0, 3, 0, 3> xwxw, rara, sqsq;
            vecX_swizzle<4, SCALAR, 0, 3, 1, 0> xwyx, ragr, sqts;
            vecX_swizzle<4, SCALAR, 0, 3, 1, 1> xwyy, ragg, sqtt;
            vecX_swizzle<4, SCALAR, 0, 3, 1, 2> xwyz, ragb, sqtp;
            vecX_swizzle<4, SCALAR, 0, 3, 1, 3> xwyw, raga, sqtq;
            vecX_swizzle<4, SCALAR, 0, 3, 2, 0> xwzx, rabr, sqps;
            vecX_swizzle<4, SCALAR, 0, 3, 2, 1> xwzy, rabg, sqpt;
            vecX_swizzle<4, SCALAR, 0, 3, 2, 2> xwzz, rabb, sqpp;
            vecX_swizzle<4, SCALAR, 0, 3, 2, 3> xwzw, raba, sqpq;
            vecX_swizzle<4, SCALAR, 0, 3, 3, 0> xwwx, raar, sqqs;
            vecX_swizzle<4, SCALAR, 0, 3, 3, 1> xwwy, raag, sqqt;
            vecX_swizzle<4, SCALAR, 0, 3, 3, 2> xwwz, raab, sqqp;
            vecX_swizzle<4, SCALAR, 0, 3, 3, 3> xwww, raaa, sqqq;
            vecX_swizzle<4, SCALAR, 1, 0, 0, 0> yxxx, grrr, tsss;
            vecX_swizzle<4, SCALAR, 1, 0, 0, 1> yxxy, grrg, tsst;
            vecX_swizzle<4, SCALAR, 1, 0, 0, 2> yxxz, grrb, tssp;
            vecX_swizzle<4, SCALAR, 1, 0, 0, 3> yxxw, grra, tssq;
            vecX_swizzle<4, SCALAR, 1, 0, 1, 0> yxyx, grgr, tsts;
            vecX_swizzle<4, SCALAR, 1, 0, 1, 1> yxyy, grgg, tstt;
            vecX_swizzle<4, SCALAR, 1, 0, 1, 2> yxyz, grgb, tstp;
            vecX_swizzle<4, SCALAR, 1, 0, 1, 3> yxyw, grga, tstq;
            vecX_swizzle<4, SCALAR, 1, 0, 2, 0> yxzx, grbr, tsps;
            vecX_swizzle<4, SCALAR, 1, 0, 2, 1> yxzy, grbg, tspt;
            vecX_swizzle<4, SCALAR, 1, 0, 2, 2> yxzz, grbb, tspp;
            vecX_swizzle<4, SCALAR, 1, 0, 2, 3> yxzw, grba, tspq;
            vecX_swizzle<4, SCALAR, 1, 0, 3, 0> yxwx, grar, tsqs;
            vecX_swizzle<4, SCALAR, 1, 0, 3, 1> yxwy, grag, tsqt;
            vecX_swizzle<4, SCALAR, 1, 0, 3, 2> yxwz, grab, tsqp;
            vecX_swizzle<4, SCALAR, 1, 0, 3, 3> yxww, graa, tsqq;
            vecX_swizzle<4, SCALAR, 1, 1, 0, 0> yyxx, ggrr, ttss;
            vecX_swizzle<4, SCALAR, 1, 1, 0, 1> yyxy, ggrg, ttst;
            vecX_swizzle<4, SCALAR, 1, 1, 0, 2> yyxz, ggrb, ttsp;
            vecX_swizzle<4, SCALAR, 1, 1, 0, 3> yyxw, ggra, ttsq;
            vecX_swizzle<4, SCALAR, 1, 1, 1, 0> yyyx, gggr, ttts;
            vecX_swizzle<4, SCALAR, 1, 1, 1, 1> yyyy, gggg, tttt;
            vecX_swizzle<4, SCALAR, 1, 1, 1, 2> yyyz, gggb, tttp;
            vecX_swizzle<4, SCALAR, 1, 1, 1, 3> yyyw, ggga, tttq;
            vecX_swizzle<4, SCALAR, 1, 1, 2, 0> yyzx, ggbr, ttps;
            vecX_swizzle<4, SCALAR, 1, 1, 2, 1> yyzy, ggbg, ttpt;
            vecX_swizzle<4, SCALAR, 1, 1, 2, 2> yyzz, ggbb, ttpp;
            vecX_swizzle<4, SCALAR, 1, 1, 2, 3> yyzw, ggba, ttpq;
            vecX_swizzle<4, SCALAR, 1, 1, 3, 0> yywx, ggar, ttqs;
            vecX_swizzle<4, SCALAR, 1, 1, 3, 1> yywy, ggag, ttqt;
            vecX_swizzle<4, SCALAR, 1, 1, 3, 2> yywz, ggab, ttqp;
            vecX_swizzle<4, SCALAR, 1, 1, 3, 3> yyww, ggaa, ttqq;
            vecX_swizzle<4, SCALAR, 1, 2, 0, 0> yzxx, gbrr, tpss;
            vecX_swizzle<4, SCALAR, 1, 2, 0, 1> yzxy, gbrg, tpst;
            vecX_swizzle<4, SCALAR, 1, 2, 0, 2> yzxz, gbrb, tpsp;
            vecX_swizzle<4, SCALAR, 1, 2, 0, 3> yzxw, gbra, tpsq;
            vecX_swizzle<4, SCALAR, 1, 2, 1, 0> yzyx, gbgr, tpts;
            vecX_swizzle<4, SCALAR, 1, 2, 1, 1> yzyy, gbgg, tptt;
            vecX_swizzle<4, SCALAR, 1, 2, 1, 2> yzyz, gbgb, tptp;
            vecX_swizzle<4, SCALAR, 1, 2, 1, 3> yzyw, gbga, tptq;
            vecX_swizzle<4, SCALAR, 1, 2, 2, 0> yzzx, gbbr, tpps;
            vecX_swizzle<4, SCALAR, 1, 2, 2, 1> yzzy, gbbg, tppt;
            vecX_swizzle<4, SCALAR, 1, 2, 2, 2> yzzz, gbbb, tppp;
            vecX_swizzle<4, SCALAR, 1, 2, 2, 3> yzzw, gbba, tppq;
            vecX_swizzle<4, SCALAR, 1, 2, 3, 0> yzwx, gbar, tpqs;
            vecX_swizzle<4, SCALAR, 1, 2, 3, 1> yzwy, gbag, tpqt;
            vecX_swizzle<4, SCALAR, 1, 2, 3, 2> yzwz, gbab, tpqp;
            vecX_swizzle<4, SCALAR, 1, 2, 3, 3> yzww, gbaa, tpqq;
            vecX_swizzle<4, SCALAR, 1, 3, 0, 0> ywxx, garr, tqss;
            vecX_swizzle<4, SCALAR, 1, 3, 0, 1> ywxy, garg, tqst;
            vecX_swizzle<4, SCALAR, 1, 3, 0, 2> ywxz, garb, tqsp;
            vecX_swizzle<4, SCALAR, 1, 3, 0, 3> ywxw, gara, tqsq;
            vecX_swizzle<4, SCALAR, 1, 3, 1, 0> ywyx, gagr, tqts;
            vecX_swizzle<4, SCALAR, 1, 3, 1, 1> ywyy, gagg, tqtt;
            vecX_swizzle<4, SCALAR, 1, 3, 1, 2> ywyz, gagb, tqtp;
            vecX_swizzle<4, SCALAR, 1, 3, 1, 3> ywyw, gaga, tqtq;
            vecX_swizzle<4, SCALAR, 1, 3, 2, 0> ywzx, gabr, tqps;
            vecX_swizzle<4, SCALAR, 1, 3, 2, 1> ywzy, gabg, tqpt;
            vecX_swizzle<4, SCALAR, 1, 3, 2, 2> ywzz, gabb, tqpp;
            vecX_swizzle<4, SCALAR, 1, 3, 2, 3> ywzw, gaba, tqpq;
            vecX_swizzle<4, SCALAR, 1, 3, 3, 0> ywwx, gaar, tqqs;
            vecX_swizzle<4, SCALAR, 1, 3, 3, 1> ywwy, gaag, tqqt;
            vecX_swizzle<4, SCALAR, 1, 3, 3, 2> ywwz, gaab, tqqp;
            vecX_swizzle<4, SCALAR, 1, 3, 3, 3> ywww, gaaa, tqqq;
            vecX_swizzle<4, SCALAR, 2, 0, 0, 0> zxxx, brrr, psss;
            vecX_swizzle<4, SCALAR, 2, 0, 0, 1> zxxy, brrg, psst;
            vecX_swizzle<4, SCALAR, 2, 0, 0, 2> zxxz, brrb, pssp;
            vecX_swizzle<4, SCALAR, 2, 0, 0, 3> zxxw, brra, pssq;
            vecX_swizzle<4, SCALAR, 2, 0, 1, 0> zxyx, brgr, psts;
            vecX_swizzle<4, SCALAR, 2, 0, 1, 1> zxyy, brgg, pstt;
            vecX_swizzle<4, SCALAR, 2, 0, 1, 2> zxyz, brgb, pstp;
            vecX_swizzle<4, SCALAR, 2, 0, 1, 3> zxyw, brga, pstq;
            vecX_swizzle<4, SCALAR, 2, 0, 2, 0> zxzx, brbr, psps;
            vecX_swizzle<4, SCALAR, 2, 0, 2, 1> zxzy, brbg, pspt;
            vecX_swizzle<4, SCALAR, 2, 0, 2, 2> zxzz, brbb, pspp;
            vecX_swizzle<4, SCALAR, 2, 0, 2, 3> zxzw, brba, pspq;
            vecX_swizzle<4, SCALAR, 2, 0, 3, 0> zxwx, brar, psqs;
            vecX_swizzle<4, SCALAR, 2, 0, 3, 1> zxwy, brag, psqt;
            vecX_swizzle<4, SCALAR, 2, 0, 3, 2> zxwz, brab, psqp;
            vecX_swizzle<4, SCALAR, 2, 0, 3, 3> zxww, braa, psqq;
            vecX_swizzle<4, SCALAR, 2, 1, 0, 0> zyxx, bgrr, ptss;
            vecX_swizzle<4, SCALAR, 2, 1, 0, 1> zyxy, bgrg, ptst;
            vecX_swizzle<4, SCALAR, 2, 1, 0, 2> zyxz, bgrb, ptsp;
            vecX_swizzle<4, SCALAR, 2, 1, 0, 3> zyxw, bgra, ptsq;
            vecX_swizzle<4, SCALAR, 2, 1, 1, 0> zyyx, bggr, ptts;
            vecX_swizzle<4, SCALAR, 2, 1, 1, 1> zyyy, bggg, pttt;
            vecX_swizzle<4, SCALAR, 2, 1, 1, 2> zyyz, bggb, pttp;
            vecX_swizzle<4, SCALAR, 2, 1, 1, 3> zyyw, bgga, pttq;
            vecX_swizzle<4, SCALAR, 2, 1, 2, 0> zyzx, bgbr, ptps;
            vecX_swizzle<4, SCALAR, 2, 1, 2, 1> zyzy, bgbg, ptpt;
            vecX_swizzle<4, SCALAR, 2, 1, 2, 2> zyzz, bgbb, ptpp;
            vecX_swizzle<4, SCALAR, 2, 1, 2, 3> zyzw, bgba, ptpq;
            vecX_swizzle<4, SCALAR, 2, 1, 3, 0> zywx, bgar, ptqs;
            vecX_swizzle<4, SCALAR, 2, 1, 3, 1> zywy, bgag, ptqt;
            vecX_swizzle<4, SCALAR, 2, 1, 3, 2> zywz, bgab, ptqp;
            vecX_swizzle<4, SCALAR, 2, 1, 3, 3> zyww, bgaa, ptqq;
            vecX_swizzle<4, SCALAR, 2, 2, 0, 0> zzxx, bbrr, ppss;
            vecX_swizzle<4, SCALAR, 2, 2, 0, 1> zzxy, bbrg, ppst;
            vecX_swizzle<4, SCALAR, 2, 2, 0, 2> zzxz, bbrb, ppsp;
            vecX_swizzle<4, SCALAR, 2, 2, 0, 3> zzxw, bbra, ppsq;
            vecX_swizzle<4, SCALAR, 2, 2, 1, 0> zzyx, bbgr, ppts;
            vecX_swizzle<4, SCALAR, 2, 2, 1, 1> zzyy, bbgg, pptt;
            vecX_swizzle<4, SCALAR, 2, 2, 1, 2> zzyz, bbgb, pptp;
            vecX_swizzle<4, SCALAR, 2, 2, 1, 3> zzyw, bbga, pptq;
            vecX_swizzle<4, SCALAR, 2, 2, 2, 0> zzzx, bbbr, ppps;
            vecX_swizzle<4, SCALAR, 2, 2, 2, 1> zzzy, bbbg, pppt;
            vecX_swizzle<4, SCALAR, 2, 2, 2, 2> zzzz, bbbb, pppp;
            vecX_swizzle<4, SCALAR, 2, 2, 2, 3> zzzw, bbba, pppq;
            vecX_swizzle<4, SCALAR, 2, 2, 3, 0> zzwx, bbar, ppqs;
            vecX_swizzle<4, SCALAR, 2, 2, 3, 1> zzwy, bbag, ppqt;
            vecX_swizzle<4, SCALAR, 2, 2, 3, 2> zzwz, bbab, ppqp;
            vecX_swizzle<4, SCALAR, 2, 2, 3, 3> zzww, bbaa, ppqq;
            vecX_swizzle<4, SCALAR, 2, 3, 0, 0> zwxx, barr, pqss;
            vecX_swizzle<4, SCALAR, 2, 3, 0, 1> zwxy, barg, pqst;
            vecX_swizzle<4, SCALAR, 2, 3, 0, 2> zwxz, barb, pqsp;
            vecX_swizzle<4, SCALAR, 2, 3, 0, 3> zwxw, bara, pqsq;
            vecX_swizzle<4, SCALAR, 2, 3, 1, 0> zwyx, bagr, pqts;
            vecX_swizzle<4, SCALAR, 2, 3, 1, 1> zwyy, bagg, pqtt;
            vecX_swizzle<4, SCALAR, 2, 3, 1, 2> zwyz, bagb, pqtp;
            vecX_swizzle<4, SCALAR, 2, 3, 1, 3> zwyw, baga, pqtq;
            vecX_swizzle<4, SCALAR, 2, 3, 2, 0> zwzx, babr, pqps;
            vecX_swizzle<4, SCALAR, 2, 3, 2, 1> zwzy, babg, pqpt;
            vecX_swizzle<4, SCALAR, 2, 3, 2, 2> zwzz, babb, pqpp;
            vecX_swizzle<4, SCALAR, 2, 3, 2, 3> zwzw, baba, pqpq;
            vecX_swizzle<4, SCALAR, 2, 3, 3, 0> zwwx, baar, pqqs;
            vecX_swizzle<4, SCALAR, 2, 3, 3, 1> zwwy, baag, pqqt;
            vecX_swizzle<4, SCALAR, 2, 3, 3, 2> zwwz, baab, pqqp;
            vecX_swizzle<4, SCALAR, 2, 3, 3, 3> zwww, baaa, pqqq;
            vecX_swizzle<4, SCALAR, 3, 0, 0, 0> wxxx, arrr, qsss;
            vecX_swizzle<4, SCALAR, 3, 0, 0, 1> wxxy, arrg, qsst;
            vecX_swizzle<4, SCALAR, 3, 0, 0, 2> wxxz, arrb, qssp;
            vecX_swizzle<4, SCALAR, 3, 0, 0, 3> wxxw, arra, qssq;
            vecX_swizzle<4, SCALAR, 3, 0, 1, 0> wxyx, argr, qsts;
            vecX_swizzle<4, SCALAR, 3, 0, 1, 1> wxyy, argg, qstt;
            vecX_swizzle<4, SCALAR, 3, 0, 1, 2> wxyz, argb, qstp;
            vecX_swizzle<4, SCALAR, 3, 0, 1, 3> wxyw, arga, qstq;
            vecX_swizzle<4, SCALAR, 3, 0, 2, 0> wxzx, arbr, qsps;
            vecX_swizzle<4, SCALAR, 3, 0, 2, 1> wxzy, arbg, qspt;
            vecX_swizzle<4, SCALAR, 3, 0, 2, 2> wxzz, arbb, qspp;
            vecX_swizzle<4, SCALAR, 3, 0, 2, 3> wxzw, arba, qspq;
            vecX_swizzle<4, SCALAR, 3, 0, 3, 0> wxwx, arar, qsqs;
            vecX_swizzle<4, SCALAR, 3, 0, 3, 1> wxwy, arag, qsqt;
            vecX_swizzle<4, SCALAR, 3, 0, 3, 2> wxwz, arab, qsqp;
            vecX_swizzle<4, SCALAR, 3, 0, 3, 3> wxww, araa, qsqq;
            vecX_swizzle<4, SCALAR, 3, 1, 0, 0> wyxx, agrr, qtss;
            vecX_swizzle<4, SCALAR, 3, 1, 0, 1> wyxy, agrg, qtst;
            vecX_swizzle<4, SCALAR, 3, 1, 0, 2> wyxz, agrb, qtsp;
            vecX_swizzle<4, SCALAR, 3, 1, 0, 3> wyxw, agra, qtsq;
            vecX_swizzle<4, SCALAR, 3, 1, 1, 0> wyyx, aggr, qtts;
            vecX_swizzle<4, SCALAR, 3, 1, 1, 1> wyyy, aggg, qttt;
            vecX_swizzle<4, SCALAR, 3, 1, 1, 2> wyyz, aggb, qttp;
            vecX_swizzle<4, SCALAR, 3, 1, 1, 3> wyyw, agga, qttq;
            vecX_swizzle<4, SCALAR, 3, 1, 2, 0> wyzx, agbr, qtps;
            vecX_swizzle<4, SCALAR, 3, 1, 2, 1> wyzy, agbg, qtpt;
            vecX_swizzle<4, SCALAR, 3, 1, 2, 2> wyzz, agbb, qtpp;
            vecX_swizzle<4, SCALAR, 3, 1, 2, 3> wyzw, agba, qtpq;
            vecX_swizzle<4, SCALAR, 3, 1, 3, 0> wywx, agar, qtqs;
            vecX_swizzle<4, SCALAR, 3, 1, 3, 1> wywy, agag, qtqt;
            vecX_swizzle<4, SCALAR, 3, 1, 3, 2> wywz, agab, qtqp;
            vecX_swizzle<4, SCALAR, 3, 1, 3, 3> wyww, agaa, qtqq;
            vecX_swizzle<4, SCALAR, 3, 2, 0, 0> wzxx, abrr, qpss;
            vecX_swizzle<4, SCALAR, 3, 2, 0, 1> wzxy, abrg, qpst;
            vecX_swizzle<4, SCALAR, 3, 2, 0, 2> wzxz, abrb, qpsp;
            vecX_swizzle<4, SCALAR, 3, 2, 0, 3> wzxw, abra, qpsq;
            vecX_swizzle<4, SCALAR, 3, 2, 1, 0> wzyx, abgr, qpts;
            vecX_swizzle<4, SCALAR, 3, 2, 1, 1> wzyy, abgg, qptt;
            vecX_swizzle<4, SCALAR, 3, 2, 1, 2> wzyz, abgb, qptp;
            vecX_swizzle<4, SCALAR, 3, 2, 1, 3> wzyw, abga, qptq;
            vecX_swizzle<4, SCALAR, 3, 2, 2, 0> wzzx, abbr, qpps;
            vecX_swizzle<4, SCALAR, 3, 2, 2, 1> wzzy, abbg, qppt;
            vecX_swizzle<4, SCALAR, 3, 2, 2, 2> wzzz, abbb, qppp;
            vecX_swizzle<4, SCALAR, 3, 2, 2, 3> wzzw, abba, qppq;
            vecX_swizzle<4, SCALAR, 3, 2, 3, 0> wzwx, abar, qpqs;
            vecX_swizzle<4, SCALAR, 3, 2, 3, 1> wzwy, abag, qpqt;
            vecX_swizzle<4, SCALAR, 3, 2, 3, 2> wzwz, abab, qpqp;
            vecX_swizzle<4, SCALAR, 3, 2, 3, 3> wzww, abaa, qpqq;
            vecX_swizzle<4, SCALAR, 3, 3, 0, 0> wwxx, aarr, qqss;
            vecX_swizzle<4, SCALAR, 3, 3, 0, 1> wwxy, aarg, qqst;
            vecX_swizzle<4, SCALAR, 3, 3, 0, 2> wwxz, aarb, qqsp;
            vecX_swizzle<4, SCALAR, 3, 3, 0, 3> wwxw, aara, qqsq;
            vecX_swizzle<4, SCALAR, 3, 3, 1, 0> wwyx, aagr, qqts;
            vecX_swizzle<4, SCALAR, 3, 3, 1, 1> wwyy, aagg, qqtt;
            vecX_swizzle<4, SCALAR, 3, 3, 1, 2> wwyz, aagb, qqtp;
            vecX_swizzle<4, SCALAR, 3, 3, 1, 3> wwyw, aaga, qqtq;
            vecX_swizzle<4, SCALAR, 3, 3, 2, 0> wwzx, aabr, qqps;
            vecX_swizzle<4, SCALAR, 3, 3, 2, 1> wwzy, aabg, qqpt;
            vecX_swizzle<4, SCALAR, 3, 3, 2, 2> wwzz, aabb, qqpp;
            vecX_swizzle<4, SCALAR, 3, 3, 2, 3> wwzw, aaba, qqpq;
            vecX_swizzle<4, SCALAR, 3, 3, 3, 0> wwwx, aaar, qqqs;
            vecX_swizzle<4, SCALAR, 3, 3, 3, 1> wwwy, aaag, qqqt;
            vecX_swizzle<4, SCALAR, 3, 3, 3, 2> wwwz, aaab, qqqp;
            vecX_swizzle<4, SCALAR, 3, 3, 3, 3> wwww, aaaa, qqqq;
        };

        explicit vecX_data(SCALAR all) { self = all; }

        explicit vecX_data(SCALAR x, SCALAR y, SCALAR z, SCALAR w) { r = x; g = y; b = z; a = w; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(vecX_data<SCALAR2, 2> xy, SCALAR z, SCALAR w) { rg = xy; b = z; a = w; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 2)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> xy, SCALAR z, SCALAR w) { rg = xy; b = z; a = w; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(SCALAR x, vecX_data<SCALAR2, 2> yz, SCALAR w) { r = x; gb = yz; a = w; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 2)
        explicit vecX_data(SCALAR x, vecX_swizzle<SIZE, SCALAR2, I...> yz, SCALAR w) { r = x; gb = yz; a = w; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(SCALAR x, SCALAR y, vecX_data<SCALAR2, 2> zw) { r = x; g = y; ba = zw; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 2)
        explicit vecX_data(SCALAR x, SCALAR y, vecX_swizzle<SIZE, SCALAR2, I...> zw) { r = x; g = y; ba = zw; }

        template<typename SCALAR2, typename SCALAR3> requires (std::convertible_to<SCALAR2, SCALAR> and std::convertible_to<SCALAR3, SCALAR>)
        explicit vecX_data(vecX_data<SCALAR2, 2> xy, vecX_data<SCALAR3, 2> zw) { rg = xy; ba = zw; }

        template<unsigned int SIZE, unsigned int SIZE2, typename SCALAR2, typename SCALAR3, unsigned int... I, unsigned int... J> requires (std::convertible_to<SCALAR2, SCALAR> and std::convertible_to<SCALAR3, SCALAR> and sizeof...(I) == 2 and sizeof...(J) == 2)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> xy, vecX_swizzle<SIZE2, SCALAR3, J...> zw) { rg = xy; ba = zw; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(vecX_data<SCALAR2, 3> xyz, SCALAR w) { rgb = xyz; a = w; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 3)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> xyz, SCALAR w) { rgb = xyz; a = w; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(SCALAR x, vecX_data<SCALAR2, 3> yzw) { r = x; gba = yzw; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 3)
        explicit vecX_data(SCALAR x, vecX_swizzle<SIZE, SCALAR2, I...> yzw) { r = x; gba = yzw; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(vecX_data<SCALAR2, 3> xyzw) { rgba = xyzw; }

        template<unsigned int SIZE, typename SCALAR2, unsigned int... I> requires (std::convertible_to<SCALAR2, SCALAR> and sizeof...(I) == 4)
        explicit vecX_data(vecX_swizzle<SIZE, SCALAR2, I...> xyzw) { self = xyzw; }

        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        explicit vecX_data(matNxM<2, 2, SCALAR2> mat2) : vecX_data(mat2[0], mat2[1]) {  };

        vecX_data& operator=(SCALAR v) {
            self = v;
            return *this;
        }

        template<unsigned int SIZE, unsigned int... I> requires (sizeof...(I) == 4)
        vecX_data& operator=(vecX_swizzle<SIZE, SCALAR, I...> v) {
            self = v;
            return *this;
        }
    };


    typedef vecX_data<float, 2> vec2;
    typedef vecX_data<double, 2> dvec2;
    typedef vecX_data<bool, 2> bvec2;
    typedef vecX_data<int, 2> ivec2;
    typedef vecX_data<uint, 2> uvec2;

    typedef vecX_data<float, 3> vec3;
    typedef vecX_data<double, 3> dvec3;
    typedef vecX_data<bool, 3> bvec3;
    typedef vecX_data<int, 3> ivec3;
    typedef vecX_data<uint, 3> uvec3;

    typedef vecX_data<float, 4> vec4;
    typedef vecX_data<double, 4> dvec4;
    typedef vecX_data<bool, 4> bvec4;
    typedef vecX_data<int, 4> ivec4;
    typedef vecX_data<uint, 4> uvec4;


    static_assert(sizeof(vecX_common<vecX_data<double, 2>, 2, double, std::make_integer_sequence<unsigned int, 2>>) == 1);
    static_assert(sizeof(vecX_swizzle<1, double, 0>) == 1 * sizeof(double));
    static_assert(sizeof(vecX_swizzle<2, char, 0, 1>) == 2 * sizeof(char));
    static_assert(sizeof(vecX_data<char, 1>) == 1*sizeof(char));
    static_assert(sizeof(vecX_data<short, 2>) == 2*sizeof(short));

    static_assert(std::convertible_to<float, vecX_data<float, 1>>);
    static_assert(std::convertible_to<int, vecX_data<float, 1>>);
    static_assert(std::convertible_to<vecX_data<float, 1>, int>);
    static_assert(!std::convertible_to<double, vecX_data<int, 2>>);
    static_assert(std::convertible_to<vecX_swizzle<3, int, 0, 1>, vecX_data<float, 2>>);
    static_assert(std::convertible_to<vecX_data<int, 2>, vecX_data<int, 2>>);
    static_assert(std::convertible_to<vecX_data<int, 2>, vecX_data<double, 2>>);

    static_assert(std::is_same_v<vecX_converter<int>::type, vecX_data<int, 1>>);
    static_assert(std::is_same_v<vecX_converter<float>::type, vecX_data<float, 1>>);
    static_assert(std::is_same_v<vecX_converter<vecX_data<int, 2>>::type, vecX_data<int, 2>>);
    static_assert(std::is_same_v<vecX_converter<vecX_swizzle<3, double, 0, 1>>::type, vecX_data<double, 2>>);


    template <unsigned int N, unsigned int M, typename SCALAR> // requires (N >= 2 and N <= 4 and M >= 2 and M <= 4)
    struct matNxM {
    public:
        union {
            SCALAR data[N][M];
            vecX_data<SCALAR, M> cols[N];
        };

        // empty (all zeros) matrix constructor
        matNxM() { for (unsigned int c = 0; c < N; c++) cols[c] = 0; }

        // diagonal matrix constructor for square matrices
        matNxM(SCALAR diag) requires (N == M) {
            for (unsigned int c = 0; c < N; c++) {
                for (unsigned int r = 0; r < M; r++) {
                    data[c][r] = (r == c) ? diag : 0;
                }
            }
        }

        // matrix constructor from another matrix, fills remaining diagonal elements with 1s
        template<unsigned int N2, unsigned int M2>
        explicit matNxM(matNxM<N2, M2, SCALAR> mat) {
            for (unsigned int c = 0; c < (N2 < N ? N2 : N); c++) {
                for (unsigned int r = 0; r < (M2 < M ? M2 : M); r++) data[c][r] = mat.data[c][r];
            }
            for (unsigned int i = (M2 < N2 ? M2 : N2); i < (M < N ? M : N); i++) data[i][i] = 1;
        }

        // matrix constructor from any vector-scalar combination that can be concatenated into an M*N-dim vector
        template<typename... Ts> requires (N*M > 1 and (vecX_compatible_dim<Ts> + ...) == N*M and (std::convertible_to<vecX_compatible_scalar<Ts>, SCALAR> and ...))
        explicit matNxM(Ts... vs) {
            vecX_data<SCALAR, N*M> vecX_conc = vecX_concat<SCALAR>(vs...);
            for (unsigned int c = 0; c < N; c++) {
                for (unsigned int r = 0; r < M; r++) {
                    data[c][r] = vecX_conc[c*M + r];
                }
            }
        }

        // implicit cast to scalar for 1x1 matrices
        operator SCALAR() const requires (N == 1 and M == 1) {
            return data[0][0];
        }

        // provides a submatrix by skipping one selected column and row (last by default)
        template<unsigned int N_skip=N-1, unsigned int M_skip=M-1> requires (M > 1 and N > 1)
        matNxM<N-1, M-1, SCALAR> submatrix() {
            auto result = matNxM<N-1, M-1, SCALAR>();
            unsigned int c = 0, r = 0;
            do {
                r = 0;
                do {
                    result[c][r] = data[(c >= N_skip) ? c + 1 : c][(r >= M_skip) ? r + 1 : r];
                    r++;
                } while(r < M - 1);
                c++;
            } while(c < N - 1);
            return result;
        }

        // overload subscript operator for array-like access to column vectors
        vecX_data<SCALAR, M>& operator [](const int c) {
            return cols[c];
        }

        // overloading << for output stream
        friend std::ostream& operator<<(std::ostream& os, const matNxM& rhs) {
            os << "(";
            for (unsigned int c = 0; c < N; c++) {
                if (c > 0) os << ", ";
                os << rhs.cols[c];
            }
            os << ")";
            return os;
        }

        // element-wise add & assign operator for matrix and scalars
        matNxM& operator +=(const SCALAR rhs) { for (unsigned int c = 0; c < N; c++) cols[c] += rhs; return *this; }

        // element-wise add & assign operator for matrix and another matrix
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM& operator +=(const matNxM<N, M, SCALAR2> rhs) { for (unsigned int c = 0; c < N; c++) cols[c] += rhs.cols[c]; return *this; }

        // element-wise add operator for matrix and scalars
        matNxM operator +(const SCALAR rhs) { auto copy = matNxM(*this); copy += rhs; return copy; }

        // element-wise add operator for matrix and another matrix
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM operator +(const matNxM<N, M, SCALAR2> rhs) { auto copy = matNxM(*this); copy += rhs; return copy; }


        // element-wise subtract & assign operator for matrix and scalars
        matNxM& operator -=(const SCALAR rhs) { for (unsigned int c = 0; c < N; c++) cols[c] -= rhs; return *this; }

        // element-wise subtract & assign operator for matrix and another matrix
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM& operator -=(const matNxM<N, M, SCALAR2> rhs) { for (unsigned int c = 0; c < N; c++) cols[c] -= rhs.cols[c]; return *this; }

        // element-wise subtract operator for matrix and scalars
        matNxM operator -(const SCALAR rhs) { auto copy = matNxM(*this); copy -= rhs; return copy; }

        // element-wise subtract operator for matrix and another matrix
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM operator -(const matNxM<N, M, SCALAR2> rhs) { auto copy = matNxM(*this); copy -= rhs; return copy; }


        // element-wise divide & assign operator for matrix and scalars
        matNxM& operator /=(const SCALAR rhs) { for (unsigned int c = 0; c < N; c++) cols[c] /= rhs; return *this; }

        // element-wise divide & assign operator for matrix and another matrix
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM& operator /=(const matNxM<N, M, SCALAR2> rhs) { for (unsigned int c = 0; c < N; c++) cols[c] /= rhs.cols[c]; return *this; }

        // element-wise divide operator for matrix and scalars
        matNxM operator /(const SCALAR rhs) { auto copy = matNxM(*this); copy /= rhs; return copy; }

        // element-wise divide operator for matrix and another matrix
        template<typename SCALAR2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM operator /(const matNxM<N, M, SCALAR2> rhs) { auto copy = matNxM(*this); copy /= rhs; return copy; }


        // element-wise multiply & assign operator for matrix and scalars
        matNxM& operator *=(const SCALAR rhs) { for (unsigned int c = 0; c < N; c++) cols[c] *= rhs; return *this; }

        // element-wise multiply operator for matrix and scalars
        matNxM operator *(const SCALAR rhs) { auto copy = matNxM(*this); copy *= rhs; return copy; }

        // inner product operator for matrix and a vector
        template<typename T> requires (vecX_compatible_dim<T> == N)
        vecX_data<SCALAR, M> operator *(const T rhs) {
            vecX_data<SCALAR, M> result(0);
            vecX_compatible_t<T> vec(rhs);
            matNxM<M, N, SCALAR> self_T = transpose(*this);
            for (unsigned int r = 0; r < M; r++) {
                result[r] = self_T[r].dot(vec);
            }
            return result;
        }

        // inner product operator for matrix and another matrix
        template<typename SCALAR2, unsigned int N2> requires (std::convertible_to<SCALAR2, SCALAR>)
        matNxM<N2, M, SCALAR> operator *(const matNxM<N2, N, SCALAR2> rhs) {
            auto result = matNxM<N2, M, SCALAR>();
            matNxM<M, N, SCALAR> self_T = transpose(*this);
            for (unsigned int c = 0; c < N2; c++) {
                for (unsigned int r = 0; r < M; r++) {
                    result[c][r] = self_T[r].dot(rhs.cols[c]);
                }
            }
            return result;
        }
    };

    typedef matNxM<2, 2, float> mat2x2;
    typedef matNxM<2, 3, float> mat2x3;
    typedef matNxM<2, 4, float> mat2x4;
    typedef matNxM<3, 2, float> mat3x2;
    typedef matNxM<3, 3, float> mat3x3;
    typedef matNxM<3, 4, float> mat3x4;
    typedef matNxM<4, 2, float> mat4x2;
    typedef matNxM<4, 3, float> mat4x3;
    typedef matNxM<4, 4, float> mat4x4;
    typedef mat2x2 mat2;
    typedef mat3x3 mat3;
    typedef mat4x4 mat4;

    static_assert(sizeof(matNxM<2, 3, float>) == 2*3*sizeof(float));



    // built-in trigonometric functions
    template<typename T> requires (vecX_compatible<T>)
    inline auto radians(T degrees) { return (degrees * M_PI) / 180; };

    template<typename T> requires (vecX_compatible<T>)
    inline auto degrees(T radians) { return (radians * 180) / M_PI; };

    template<typename T> requires (vecX_compatible<T>)
    inline auto sin(T angle) { return vecX_compatible_t<T>(angle).apply(::sin); };

    /*template<typename SCALAR, unsigned int DIM>
    vecX_data<SCALAR, DIM> sin(vecX_data<SCALAR, DIM> angle) { return angle.apply(::sin); }
    double sin(double angle) { return ::sin(angle); }*/

    template<typename T> requires (vecX_compatible<T>)
    inline auto cos(T angle) { return vecX_compatible_t<T>(angle).apply(::cos); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto tan(T angle) { return vecX_compatible_t<T>(angle).apply(::tan); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto asin(T x) { return vecX_compatible_t<T>(x).apply(::asin); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto acos(T x) { return vecX_compatible_t<T>(x).apply(::acos); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto atan(T y_over_x) { return vecX_compatible_t<T>(y_over_x).apply(::atan); };

    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto atan(T1 y, T2 x) { return vecX_compatible_t<T1>(y).apply(::atan2, x); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto sinh(T x) { return vecX_compatible_t<T>(x).apply(::sinh); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto cosh(T x) { return vecX_compatible_t<T>(x).apply(::cosh); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto tanh(T x) { return vecX_compatible_t<T>(x).apply(::tanh); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto asinh(T x) { return vecX_compatible_t<T>(x).apply(::asinh); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto acosh(T x) { return vecX_compatible_t<T>(x).apply(::acosh); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto atanh(T x) { return vecX_compatible_t<T>(x).apply(::atanh); };

    // built-in exponential functions
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto pow(T1 x, T2 y) { return vecX_compatible_t<T1>(x).apply(::pow, y); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto exp(T x) { return vecX_compatible_t<T>(x).apply(::exp); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto log(T x) { return vecX_compatible_t<T>(x).apply(::log); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto exp2(T x) { return vecX_compatible_t<T>(x).apply(::exp2); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto log2(T x) { return vecX_compatible_t<T>(x).apply(::log2); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto sqrt(T x) { return vecX_compatible_t<T>(x).apply(::sqrt); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto inversesqrt(T x) { return 1.0 / sqrt(x); };

    // built-in common functions
    template<typename T> requires (vecX_compatible<T>)
    inline auto abs(T x) { return vecX_compatible_t<T>(x).apply(::fabs); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto sign(T x) { return vecX_compatible_t<T>(x).apply([](double x) -> double { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto floor(T x) { return vecX_compatible_t<T>(x).apply(::floor); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto trunc(T x) { return vecX_compatible_t<T>(x).apply(::trunc); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto round(T x) { return vecX_compatible_t<T>(x).apply(::round); };

    //template<typename T> requires (vecX_compatible<T>)
    //TODO auto roundEven(T x) { return vecX_compatible_t<T>(x).apply(::roundEven); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto ceil(T x) { return vecX_compatible_t<T>(x).apply(::ceil); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto fract(T x) { return x - floor(x); };

    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto mod(T1 x, T2 y) { return vecX_compatible_t<T1>(x).apply(::fmod, y); };

    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto modf(T1 x, T2 i) { return vecX_compatible_t<T1>(x).apply_pairs(::modf, i); };

    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto min(T1 x, T2 y) { return vecX_compatible_t<T1>(x).apply([](double x, double y) -> double { return (x < y) ? x : y; }, y); };

    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto max(T1 x, T2 y) { return vecX_compatible_t<T1>(x).apply([](double x, double y) -> double { return (x > y) ? x : y; }, y); };

    template<typename T1, typename T2, typename T3> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible<T3>)
    inline auto clamp(T1 x, T2 minVal, T3 maxVal) { return min(max(x, minVal), maxVal); };

    template<typename T1, typename T2, typename T3> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible<T3>)
    inline auto mix(T1 x, T2 y, T3 a) { return x * (1.0 - a) + y * a; };

    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto step(T1 edge, T2 x) { return vecX_compatible_t<T1>(x).apply([](double x, double edge) -> double { return (x < edge) ? 0 : 1; }, edge); };

    template<typename T1, typename T2, typename T3> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible<T3>)
    inline auto smoothstep(T1 edge0, T2 edge1, T3 x) { auto t = clamp((x - edge0) / (edge1 - edge0), 0, 1); return t * t * (3 - 2 * t); };

    template<typename T> requires (vecX_compatible<T>)
    inline auto isnan(T x) { return vecX_compatible_t<T>(x).apply(::isnanf);  };

    template<typename T> requires (vecX_compatible<T>)
    inline auto isinf(T x) { return vecX_compatible_t<T>(x).apply(::isinff);  };

    // built-in geometric functions
    // built-in function for L2 length of a vector
    template<typename T> requires (vecX_compatible<T>)
    inline auto length(T x) { return vecX_compatible_t<T>(x).length();  };

    // built-in function for L2 distance between two vectors
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto distance(T1 p0, T2 p1) { return (p0 - p1).length();  };

    // built-in function for inner (dot) product of two vectors
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto dot(T1 x, T2 y) { return vecX_compatible_t<T1>(x).dot(y);  };

    // built-in function for a cross product of two 3D vectors
    inline vec3 cross(vec3 x, vec3 y) { return vec3(x[1] * y[2] - y[1] - x[2], x[2] * y[0] - y[2] * x[0], x[0] * y[1] - y[0] * x[1]); };

    // built-in function for a normalization of a vector
    template<typename T> requires (vecX_compatible<T>)
    inline auto normalize(T x) { auto v = vecX_compatible_t<T>(x); return v / v.length();  };

    // built-in function returning +N or -N based on dot product of vector I with referential direction Nref
    template<typename T1, typename T2, typename T3> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible<T3>)
    inline auto faceforward(T1 N, T2 I, T3 Nref) { return dot(Nref, I) < 0 ? N : -N; };

    // built-in function for reflection vector given incident vector I and surface orientation N
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto reflect(T1 I, T2 N) { return I - 2 * dot(N, I) * N;  };

    // built-in function for refraction vector given incident vector I and surface normal N
    template<typename T1, typename T2, typename T3> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible<T3>)
    inline auto refract(T1 I, T2 N, float eta) { auto k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I)); return (k < 0.0) ? vecX_compatible_t<T1>(0.0) : eta * I - (eta * dot(N, I) + sqrt(k)) * N; };

    // built-in matrix functions
    // built-in function for matrix transposition
    template <unsigned int N, unsigned int M, typename SCALAR>
    inline matNxM<M, N, SCALAR> transpose(matNxM<N, M, SCALAR> m) {
        auto ret = matNxM<M, N, SCALAR>();
        for (unsigned int c = 0; c < N; c++) {
            for (unsigned int r = 0; r < M; r++) {
                ret[r][c] = m[c][r];
            }
        }
        return ret;
    }

    // built-in function for determinant of a square matrix (1x1 case)
    template <unsigned int N, typename SCALAR> requires (N == 1)
    inline SCALAR determinant(matNxM<N, N, SCALAR> m) {
        return m[0][0];
    }

    // built-in function for determinant of a square matrix
    template <unsigned int N, typename SCALAR>
    inline SCALAR determinant(matNxM<N, N, SCALAR> m) {
        auto tmp = matNxM<N, N, SCALAR>();
        int s = 1;
        SCALAR det = 0;
        unsigned int i, j, p, q, k;
        for (k = 0; k < N; k++) {
            p = 0;
            q = 0;
            for (i = 0; i < N ; i++) {
                for (j = 0; j < N; j++) {
                    tmp[i][j] = 0;
                    if (i != 0 && j != k) {
                        tmp[p][q] = m[i][j];
                        if (q < N - 2) {
                            q++;
                        } else {
                            q = 0;
                            p++;
                        }
                    }
                }
            }
            det += s * m[0][k] * determinant(tmp.submatrix());
            s *= -1;
        }
        return det;
    }


    // built-in function for inverse of a square matrix
    template <unsigned int N, typename SCALAR>
    inline matNxM<N, N, SCALAR> inverse(matNxM<N, N, SCALAR> m) {
        auto tmp = matNxM<N, N, SCALAR>();
        auto adj = matNxM<N, N, SCALAR>();
        int p, q, s, t, i, j;
        for (q = 0; q < N; q++) {
            for (p = 0; p < N; p++) {
                s = 0;
                t = 0;
                for (i = 0; i < N; i++) {
                    for (j = 0; j < N; j++) {
                        if (i != q && j != p) {
                            tmp[s][t] = m[i][j];
                            if (t < N - 2) {
                                t++;
                            } else {
                                t = 0;
                                s++;
                            }
                        }
                    }
                }
                adj[p][q] = pow(-1, q + p) * determinant(tmp.submatrix());
            }
        }
        return adj / determinant(m);
    }

    // built-in function for matrix element-wise multiplication
    template <unsigned int N, unsigned int M, typename SCALAR>
    inline matNxM<N, M, SCALAR> matrixCompMult(matNxM<N, M, SCALAR> x, matNxM<N, M, SCALAR> y) {
        auto ret = matNxM<N, M, SCALAR>();
        for (unsigned int c = 0; c < N; c++) {
            for (unsigned int r = 0; r < M; r++) {
                ret[c][r] = x[c][r] * y[c][r];
            }
        }
        return ret;
    }

    // built-in function for vector outer product
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2>)
    inline auto outerProduct(T1 c, T2 r) {
        const unsigned int N = vecX_compatible_dim<T2>;
        const unsigned int M = vecX_compatible_dim<T1>;
        auto lhs = vecX_compatible_t<T1>(c);
        auto rhs = vecX_compatible_t<T2>(r);
        auto ret = matNxM<N, M, vecX_compatible_scalar<T1>>();
        for (unsigned int i_c = 0; i_c < N; i_c++) {
            for (unsigned int i_r = 0; i_r < M; i_r++) {
                ret[i_c][i_r] = lhs[i_r] * rhs[i_c];
            }
        }
        return ret;
    }

    // built-in vector relational functions
    // built-in vector element-wise < operator
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto lessThan(T1 x, T2 y) {
        const unsigned int N = vecX_compatible_dim<T1>;
        auto ret = vecX_data<bool, N>();
        auto lhs = vecX_compatible_t<T1>(x);
        auto rhs = vecX_compatible_t<T2>(y);
        for (unsigned int i = 0; i < N; i++) {
            ret[i] = lhs[i] < rhs[i];
        }
        return ret;
    }

    // built-in vector element-wise <= operator
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto lessThanEqual(T1 x, T2 y) {
        const unsigned int N = vecX_compatible_dim<T1>;
        auto ret = vecX_data<bool, N>();
        auto lhs = vecX_compatible_t<T1>(x);
        auto rhs = vecX_compatible_t<T2>(y);
        for (unsigned int i = 0; i < N; i++) {
            ret[i] = lhs[i] <= rhs[i];
        }
        return ret;
    }

    // built-in vector element-wise > operator
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto greaterThan(T1 x, T2 y) {
        const unsigned int N = vecX_compatible_dim<T1>;
        auto ret = vecX_data<bool, N>();
        auto lhs = vecX_compatible_t<T1>(x);
        auto rhs = vecX_compatible_t<T2>(y);
        for (unsigned int i = 0; i < N; i++) {
            ret[i] = lhs[i] > rhs[i];
        }
        return ret;
    }

    // built-in vector element-wise >= operator
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto greaterThanEqual(T1 x, T2 y) {
        const unsigned int N = vecX_compatible_dim<T1>;
        auto ret = vecX_data<bool, N>();
        auto lhs = vecX_compatible_t<T1>(x);
        auto rhs = vecX_compatible_t<T2>(y);
        for (unsigned int i = 0; i < N; i++) {
            ret[i] = lhs[i] >= rhs[i];
        }
        return ret;
    }

    // built-in vector element-wise == operator
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto equal(T1 x, T2 y) {
        const unsigned int N = vecX_compatible_dim<T1>;
        auto ret = vecX_data<bool, N>();
        auto lhs = vecX_compatible_t<T1>(x);
        auto rhs = vecX_compatible_t<T2>(y);
        for (unsigned int i = 0; i < N; i++) {
            ret[i] = lhs[i] == rhs[i];
        }
        return ret;
    }

    // built-in vector element-wise != operator
    template<typename T1, typename T2> requires (vecX_compatible<T1> and vecX_compatible<T2> and vecX_compatible_dim<T1> == vecX_compatible_dim<T2>)
    inline auto notEqual(T1 x, T2 y) {
        const unsigned int N = vecX_compatible_dim<T1>;
        auto ret = vecX_data<bool, N>();
        auto lhs = vecX_compatible_t<T1>(x);
        auto rhs = vecX_compatible_t<T2>(y);
        for (unsigned int i = 0; i < N; i++) {
            ret[i] = lhs[i] != rhs[i];
        }
        return ret;
    }

    // built-in boolean vector check if any component is true
    template<typename T> requires (vecX_compatible<T> and std::same_as<vecX_compatible_scalar<T>, bool>)
    inline bool any(T x) {
        auto vec = vecX_compatible_t<T>(x);
        for (unsigned int i = 0; i < vecX_compatible_dim<T>; i++) {
            if (vec[i] == true) return true;
        }
        return false;
    }

    // built-in boolean vector check if all components are true
    template<typename T> requires (vecX_compatible<T> and std::same_as<vecX_compatible_scalar<T>, bool>)
    inline bool all(T x) {
        auto vec = vecX_compatible_t<T>(x);
        for (unsigned int i = 0; i < vecX_compatible_dim<T>; i++) {
            if (vec[i] == false) return false;
        }
        return true;
    }

    // built-in boolean vector negation
    template<typename T> requires (vecX_compatible<T> and std::same_as<vecX_compatible_scalar<T>, bool>)
    inline auto _not(T x) {
        auto ret = vecX_compatible_t<T>(x);
        for (unsigned int i = 0; i < vecX_compatible_dim<T>; i++) {
            ret[i] = !ret[i];
        }
        return ret;
    }

    template<unsigned int NUM_COMPONENTS, typename COMPONENT_TYPE>
    struct image_unit {
        using type = COMPONENT_TYPE;
    public:
        COMPONENT_TYPE values[NUM_COMPONENTS];
    } __attribute__((packed));

    typedef image_unit<4, gles::GLfloat> rgba32f;
    typedef image_unit<4, gles::GLhalf> rgba16f;
    typedef image_unit<4, gles::GLuint> rgba32ui;
    typedef image_unit<4, gles::GLushort> rgba16ui;
    typedef image_unit<4, gles::GLubyte> rgba8ui;
    typedef image_unit<4, gles::GLint> rgba32i;
    typedef image_unit<4, gles::GLshort> rgba16i;
    typedef image_unit<4, gles::GLbyte> rgba8i;

    typedef image_unit<2, gles::GLfloat> rg32f;
    typedef image_unit<2, gles::GLhalf> rg16f;
    typedef image_unit<2, gles::GLuint> rg32ui;
    typedef image_unit<2, gles::GLushort> rg16ui;
    typedef image_unit<2, gles::GLubyte> rg8ui;
    typedef image_unit<2, gles::GLint> rg32i;
    typedef image_unit<2, gles::GLshort> rg16i;
    typedef image_unit<2, gles::GLbyte> rg8i;

    typedef image_unit<1, gles::GLfloat> r32f;
    typedef image_unit<1, gles::GLhalf> r16f;
    typedef image_unit<1, gles::GLuint> r32ui;
    typedef image_unit<1, gles::GLushort> r16ui;
    typedef image_unit<1, gles::GLubyte> r8ui;
    typedef image_unit<1, gles::GLint> r32i;
    typedef image_unit<1, gles::GLshort> r16i;
    typedef image_unit<1, gles::GLbyte> r8i;

    template<typename VEC_SCALAR, unsigned int DIM, typename UNIT, uniform_access ACCESS>
    class imageXD {
    public:
        vecX_data<int, DIM> dimensions = vecX_data<int, DIM>(0);
        UNIT* data = NULL;

        imageXD(shader_instance_cpu* instance_ptr, const char* name) {
            instance_ptr->add_variable(this, name);
        };

        template<typename... T> requires ((std::same_as<T, int> and ...) and sizeof...(T) == DIM)
        void bind(UNIT* data_ptr, T... dim_sizes) {
            data = data_ptr;
            dimensions = vecX_data<int, DIM>(dim_sizes...);
        }

        // overload subscript operator for array-like access to individual image elements
        UNIT* operator [](vecX_data<int, DIM> P) {
            if (data == NULL) return NULL;
            unsigned int loc1d = 0;
            unsigned int coord_lap = 1;
            for (unsigned int i = 0; i < DIM; i++) {
                if (P[i] < 0 || P[i] >= dimensions[i]) return NULL;
                loc1d += P[i] * coord_lap;
                coord_lap *= dimensions[i];
            }
            return &(data[loc1d]);
        }
    };

    template<typename UNIT, uniform_access ACCESS=readwrite>
    using image2D = imageXD<gles::GLfloat, 2, UNIT, ACCESS>;
    template<typename UNIT, uniform_access ACCESS=readwrite>
    using iimage2D = imageXD<gles::GLint, 2, UNIT, ACCESS>;
    template<typename UNIT, uniform_access ACCESS=readwrite>
    using uimage2D = imageXD<gles::GLuint, 2, UNIT, ACCESS>;

    template<typename UNIT, uniform_access ACCESS=readwrite>
    using image3D = imageXD<gles::GLfloat, 3, UNIT, ACCESS>;
    template<typename UNIT, uniform_access ACCESS=readwrite>
    using iimage3D = imageXD<gles::GLint, 3, UNIT, ACCESS>;
    template<typename UNIT, uniform_access ACCESS=readwrite>
    using uimage3D = imageXD<gles::GLuint, 3, UNIT, ACCESS>;

    template<typename VEC_SCALAR, unsigned int DIM, typename UNIT, uniform_access ACCESS>
    inline vecX_data<int, DIM> imageSize(imageXD<VEC_SCALAR, DIM, UNIT, ACCESS>& image) {
        return image.dimensions;
    }

    template<typename VEC_SCALAR, unsigned int DIM, typename UNIT, uniform_access ACCESS>
    inline vecX_data<VEC_SCALAR, 4> imageLoad(imageXD<VEC_SCALAR, DIM, UNIT, ACCESS>& image, vecX_data<int, DIM> P) {
        const unsigned int num_components = sizeof(UNIT) / sizeof(typename UNIT::type);
        UNIT* unit = image[P];
        if (unit != NULL) {
            if constexpr (num_components == 1) {
                return vecX_data<VEC_SCALAR, 4>(unit->values[0], 0, 0, 0);
            } else if constexpr (num_components == 2) {
                return vecX_data<VEC_SCALAR, 4>(unit->values[0], unit->values[1], 0, 0);
            } else if constexpr (num_components == 3) {
                return vecX_data<VEC_SCALAR, 4>(unit->values[0], unit->values[1], unit->values[2], 0);
            } else if constexpr (num_components == 4) {
                return vecX_data<VEC_SCALAR, 4>(unit->values[0], unit->values[1], unit->values[2], unit->values[3]);
            }
        }
        return vecX_data<VEC_SCALAR, 4>(0);
    }

    template<typename VEC_SCALAR, unsigned int DIM, typename UNIT, uniform_access ACCESS>
    inline void imageStore(imageXD<VEC_SCALAR, DIM, UNIT, ACCESS>& image, vecX_data<int, DIM> P, vecX_data<VEC_SCALAR, 4> data) {
        const unsigned int num_components = sizeof(UNIT) / sizeof(typename UNIT::type);
        auto unit_vec = (vecX_data<typename UNIT::type, num_components>*) image[P];
        if (unit_vec != NULL) {
            if constexpr (num_components == 1) {
                unit_vec->x = data.x;
            } else if constexpr (num_components == 2) {
                unit_vec->xy = data.xy;
            } else if constexpr (num_components == 3) {
                unit_vec->xyz = data.xyz;
            } else if constexpr (num_components == 4) {
                *unit_vec = data;
            }
        }
    }

    template<mem_layout MEM, typename UNIT>
    class buffer {
    public:
        UNIT* data = NULL;
        unsigned int num_units = 0;

        buffer(shader_instance_cpu* instance_ptr, const char* name) {
            instance_ptr->add_variable(this, name);
        };

        void bind(UNIT* data_ptr, unsigned int len) {
            data = data_ptr;
            num_units = len;
        }

        unsigned int length() {
            return num_units;
        }

        UNIT& operator [](int i) {
            return data[i];
        }
    };
};


namespace shdr {
}


#endif //SHDR_SHDR_H
