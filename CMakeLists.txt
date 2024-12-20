cmake_minimum_required(VERSION 3.25)
project(shdr)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_compile_options(-O3)
#add_compile_options(-fconcepts) #:: for GCC-9 C++20 'requires' expression

# Find all required libraries for linking
file(GLOB RENDER_DEVICES "/dev/dri/renderD*" "/dev/dri/card*")
string(JOIN "," RENDER_DEVICES_STR ${RENDER_DEVICES})
message("Render devices and cards found: ${RENDER_DEVICES_STR}")
add_compile_definitions(RENDER_DEVICES="${RENDER_DEVICES_STR}")

message("[Library search start]")

message("[Including GLES library]")
include(cmake/FindOpenGLES.cmake)
message("-- Found version: " ${GLES_VERSION_STRING})
message("-- Shared object location: " ${GLES_LIBRARIES})
message("-- Include directory location: " ${GLES_INCLUDE_DIRS})

message("[Including EGL library]")
include(cmake/FindEGL.cmake)
message("-- Found version: " ${EGL_VERSION_STRING})
message("-- Shared object location: " ${EGL_LIBRARIES})
message("-- Include directory location: " ${EGL_INCLUDE_DIRS})

message("[Including GBM library]")
include(cmake/FindGBM.cmake)
message("-- Found version: " ${GBM_VERSION})
message("-- Shared object location: " ${GBM_LIBRARIES})
message("-- Include directory location: " ${GBM_INCLUDE_DIRS})

message("[Completed all cmake includes]")

message("[Library search end]")

# Include required library headers
include_directories (inc ${EGL_INCLUDE_DIRS} ${GBM_INCLUDE_DIRS} ${GLES_INCLUDE_DIRS})

message("[Generating targets for GLSL files]")
# Find all GLSL shader sources and add object targets to be appended to the final library binary
file (GLOB_RECURSE SHADER_FILES_PATHS CONFIGURE_DEPENDS "*.comp" "*.vert" "*.frag" "*.tess" "*.geom")
set (SHADER_OBJECTS)
set (SHADER_LIBS)
foreach(SHADER_FILE_PATH ${SHADER_FILES_PATHS})
    file(RELATIVE_PATH SHADER_FILE ${CMAKE_SOURCE_DIR} ${SHADER_FILE_PATH})
    get_filename_component(SHADER_LIB_NAME ${SHADER_FILE_PATH} NAME)
    string(REPLACE "." "_" SHADER_LIB_NAME ${SHADER_LIB_NAME})
    list(LENGTH SHADER_LIBS SHADER_LIBS_COUNT)
    message("(#" ${SHADER_LIBS_COUNT} ") Static library '" ${SHADER_LIB_NAME} "' from shader source file '" ${SHADER_FILE} "' (path: '" ${SHADER_FILE_PATH} "')")

    set(SHADER_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_LIB_NAME}.o")
    add_custom_command(
            OUTPUT ${SHADER_OBJ}
            COMMAND ld -r -b binary -o ${SHADER_OBJ} ${SHADER_FILE}
            DEPENDS ${SHADER_FILE}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM
    )
    list(APPEND SHADER_OBJECTS ${SHADER_OBJ})

    add_library(${SHADER_LIB_NAME} STATIC shdr.cpp ${SHADER_FILE} ${SHADER_OBJ})
    set_source_files_properties(${SHADER_FILE} TARGET_DIRECTORY ${SHADER_LIB_NAME} PROPERTIES LANGUAGE CXX)
    target_compile_definitions(${SHADER_LIB_NAME} PRIVATE SHADER_LIB_NAME=${SHADER_LIB_NAME})
    list(APPEND SHADER_LIBS ${SHADER_LIB_NAME})
endforeach()


message("[Generating other targets]")
add_executable(test_glsl test_glsl.cpp)

add_executable(test_conv2d test_conv2d.cpp)
target_link_libraries(test_conv2d conv2d_comp)

message("[CMakeLists end]")
