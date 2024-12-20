#include "shdr.h"

#include <iostream>



using namespace glsl;

extern char EXTERN_SHADER_SRC_VAR[];

const char* shader_lib_name = SHADER_LIB_NAME_STR;

shader_instance_cpu CPU_INSTANCE_VAR;

extern MAIN_SHADER_FUNCTION;

//#define layout(...) __VA_ARGS__
#define L(...) __VA_ARGS__

//layout(rgba8ui, binding = 1) writeonly uniform highp uimage2D output_image2d;
//layout(std430, binding = 2) buffer kernel_ssbo { uint kernel_ssbo_data[]; };

void CPU_WRAPPER_FUNCTION() {
    auto gl_globalInvocationID = uvec3(0);
    MAIN_SHADER_FUNCTION_CALL(gl_globalInvocationID);
}

