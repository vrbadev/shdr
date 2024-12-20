#include "shdr.h"

#include <iostream>

int main() {
    glsl::ivec3 v1(2, 3, -9);
    glsl::vec2 v2 = glsl::vec2(glsl::vec4(-5, 6, 7, 9).zww);
    glsl::bvec4 v3(glsl::vec2(0.0, 7.7), glsl::ivec2(1, 0));
    glsl::vec4 v4 = vecX_concat<float>(v2.xy, glsl::vec2(8).xx);

    std::cout << "v1: " << v1 << std::endl;
    std::cout << "v2: " << v2 << std::endl;
    std::cout << "v3: " << v3 << std::endl;
    std::cout << "v4: " << v4 << std::endl;

    v1.zxy = v1.y + 4;
    std::cout << "v1: " << v1 << std::endl;
    v2.xy /= 8.0f * glsl::vec2(v1.xy) - (v2.yy + 1) * v4.w;

    std::cout << "vec: " << v2 << std::endl;
    std::cout << "x: " << v2.x << std::endl;
    std::cout << "y: " << v2.y << std::endl;
    std::cout << "xx: " << v2.xx << std::endl;
    std::cout << "xy: " << v2.xy << std::endl;
    std::cout << "yx: " << v2.yx << std::endl;
    std::cout << "yy: " << v2.yy << std::endl;

    std::cout << "sub: " << v2.yx[0] << std::endl;
    std::cout << "conv: " << (glsl::vecX_swizzle<2, float, 1, 1>) v2.xy << std::endl;

    std::cout << "concat: " << vecX_concat<float>(v2, glsl::vec4(5,6,7,8), v4.yy) << std::endl;

    glsl::vec2 test(v2);
    test.yx = v2.xy * 5.0f;
    std::cout << "test: " << test << std::endl;

    glsl::mat2x3 m1(glsl::vec2(1.0f, v3.x), v1.z, 1, 8, 8);
    std::cout << "m1: " << m1 << std::endl;
    std::cout << "m2: " << glsl::mat2(1, 2, 3, 4) << std::endl;
    std::cout << "sub: " << glsl::mat2x3(1, 2, 3, 4, 5, 6).submatrix() << std::endl;

    std::cout << "radians: " << glsl::radians(30.0f) << std::endl;
    std::cout << "radians: " << glsl::radians(glsl::vec2(90, 180).yx) << std::endl;

    std::cout << "vec1: " << glsl::vecX_data<int, 1>(10) << std::endl;


    std::cout << "sin: " << glsl::sin(glsl::vec3(M_PI, 6, 15.5f)) << std::endl;
    std::cout << "sin: " << glsl::sin(1.0f) << std::endl;
    std::cout << "sin: " << glsl::sin(glsl::vecX_data<double, 2>(2).y) << std::endl;
    std::cout << "asin: " << glsl::asin(glsl::vecX_data<float, 1>(0.1)) << std::endl;

    std::cout << "atan2: " << glsl::degrees(glsl::atan(glsl::vec3(0, 3, -8888), glsl::vec3(1, 3, 0.01))) << std::endl;

    std::cout << "fract: " << glsl::fract(4.12121) << std::endl;


    std::cout << "mat vec mul: " << (glsl::mat3(1, 2, 3, 4, 5, 6, 7, 8, 9) * glsl::vec3(5.5, -8.5, 9.5)) << std::endl;
    std::cout << "mat mat mul: " << (glsl::mat3x2(1, 2, 3, 4, 5, 6) * glsl::mat3(1)) << std::endl;

    std::cout << "mat det: " << (glsl::determinant(glsl::mat3(1, -1, 4, 2, 3, 1, 4, 0, 0))) << std::endl;
    std::cout << "mat inv: " << (glsl::inverse(glsl::mat3(3, 1, 3, 5, 5, 9, 2, 8, 2))) << std::endl;

    std::cout << "outer: " << (glsl::outerProduct(glsl::vec3(1, 2, 3), glsl::vec2(4, 5))) << std::endl;
}
