#!/usr/bin/env python3
"""Exercise native GLSL frontend semantics, diagnostics and C++11 adapter execution."""

import argparse # Read the build or test command options.
import importlib.util # Load the local frontend directly for these regression checks.
from pathlib import Path # Read shader inputs and write generated adapter paths.
import subprocess # Compile and execute generated adapters or validate GLSL text.
import tempfile # Keep temporary generated files outside the project source tree.
import unittest # Describe compiler behavior as executable regression checks.

ROOT = Path(__file__).resolve().parents[1] # Compute ROOT for the following operation.
SPEC = importlib.util.spec_from_file_location( # Compute SPEC for the following operation.
    "shdr_compile_shader", ROOT / "scripts" / "compile_shader.py" # Supply shdr compile shader for the continued expression.
) # Finish this expression or collection.
COMPILER = importlib.util.module_from_spec(SPEC) # Compute COMPILER for the following operation.
SPEC.loader.exec_module(COMPILER) # Apply exec module to these arguments.
CPP_COMPILER = None # Compute CPP COMPILER for the following operation.
GLSLANG_VALIDATOR = None # Compute GLSLANG VALIDATOR for the following operation.
EGL_FLAGS = [] # Compute EGL FLAGS for the following operation.


class FrontendTests(unittest.TestCase): # Group these regression checks.
    """Check interface-validation diagnostics during adapter generation."""

    def test_comments_and_literals_do_not_declare_interfaces(self): # Verify interface parsing preserves comments and literals.
        source = '''
// layout(local_size_x=0) in; shared atomicAdd barrier
/* uniform sampler2D fake; layout(std140,binding=0) buffer Broken {}; */
layout(local_size_x=2, local_size_y=3) in;
const char* diagnostic = "uniform sampler2D fake; atomicAdd barrier .rgba";
void main() {}
'''
        cpu, gpu, local = COMPILER.transform(source, "lexical_kernel") # Apply transform to these arguments.
        self.assertEqual(local, [2, 3, 1]) # Check the expected frontend result or execution status.
        self.assertIn('"uniform sampler2D fake; atomicAdd barrier .rgba"', cpu) # Check the expected frontend result or execution status.
        self.assertEqual( # Check the expected frontend result or execution status.
            gpu, # Supply gpu for the continued expression.
            source.replace( # Apply replace to these arguments.
                "layout(local_size_x=2, local_size_y=3) in;",
                "layout(local_size_x=2,local_size_y=3,local_size_z=1) in;",
            ), # Finish this expression or collection.
        ) # Finish this expression or collection.
        masked = COMPILER.mask_noncode(source) # Compute masked for the following operation.
        self.assertEqual(len(masked), len(source)) # Check the expected frontend result or execution status.
        self.assertEqual( # Check the expected frontend result or execution status.
            [i for i, c in enumerate(masked) if c == "\n"], # Apply enumerate to these arguments.
            [i for i, c in enumerate(source) if c == "\n"], # Apply enumerate to these arguments.
        ) # Finish this expression or collection.
        for annotation in ('// Include CPU types.', '/* Include CPU types. */'): # Process each annotation in this sequence.
            annotated = ( # Compute annotated for the following operation.
                '#ifdef __cplusplus\n#include "shdr.h" '
                + annotation # Supply annotation for the continued expression.
                + '\n#endif\nvoid main(){}'
            ) # Finish this expression or collection.
            _, gpu, _ = COMPILER.transform(annotated, "annotated_include_kernel") # Apply transform to these arguments.
            self.assertNotIn('#include', gpu) # Check the expected frontend result or execution status.

    def test_rejects_unsupported_declarations(self): # Verify declaration validation diagnostics.
        base = "layout(local_size_x=1) in;\n{}\nvoid main() {{}}\n" # Compute base for the following operation.
        cases = [ # Compute cases for the following operation.
            ("layout(std140,binding=0) buffer B {float values[];};", "std430"), # Apply layout to these arguments.
            ("layout(std430,binding=0) buffer B {vec3 values[];};", "scalar/vec2/vec4"), # Apply layout to these arguments.
            ( # Finish this expression or collection.
                "layout(std430,binding=0) buffer B {float a; float values[];};",
                "one runtime-sized",
            ), # Finish this expression or collection.
            ( # Finish this expression or collection.
                "layout(std430,binding=0) readonly writeonly buffer B {float values[];};",
                "both readonly",
            ), # Finish this expression or collection.
            ( # Finish this expression or collection.
                "layout(std430,binding=0) buffer A {float a[];};\n"
                "layout(std430,binding=0) buffer B {float b[];};",
                "duplicate buffer binding",
            ), # Finish this expression or collection.
            ("uniform float value; uniform uint value;", "duplicate resource"), # Supply uniform for the continued expression.
            ("uniform sampler2D source;", "unsupported uniform type"), # Supply uniform for the continued expression.
            ("layout(r32f,binding=0) uniform iimage2D source;", "type and format"), # Apply layout to these arguments.
            ("layout(rgba32f,binding=0) uniform image2D source;", "RGBA images"), # Apply layout to these arguments.
            ( # Finish this expression or collection.
                "layout(rgba8,binding=0) readonly uniform image2D source;",
                "supported r32/rgba32",
            ), # Finish this expression or collection.
            ( # Finish this expression or collection.
                "layout(r32f,binding=0) readonly uniform image2D a;\n"
                "layout(r32f,binding=0) writeonly uniform image2D b;",
                "duplicate image binding",
            ), # Finish this expression or collection.
            ("shared float values[16];", "outside the CPU portable subset"), # Supply shared for the continued expression.
            ("void wait_for_group(){barrier();}", "outside the CPU portable subset"), # Apply wait for group to these arguments.
            ( # Finish this expression or collection.
                "void atomic_update(){atomicAdd(counter,1u);}",
                "outside the CPU portable subset",
            ), # Finish this expression or collection.
        ] # Finish this expression or collection.
        for declaration, message in cases: # Process each declaration, message in this sequence.
            with self.subTest(declaration=declaration): # Keep this temporary resource alive throughout the following block.
                with self.assertRaisesRegex(COMPILER.ShaderError, message): # Keep this temporary resource alive throughout the following block.
                    COMPILER.generate(base.format(declaration), "invalid_kernel") # Apply generate to these arguments.

        for source, message in [ # Process each source, message in this sequence.
            ("layout(local_size_x=0) in; void main(){}", "positive"), # Apply layout to these arguments.
            ( # Finish this expression or collection.
                "layout(local_size_x=1,local_size_x=2) in; void main(){}",
                "duplicate layout",
            ), # Finish this expression or collection.
            ( # Finish this expression or collection.
                "layout(local_size_x=1, local_size_x = 2) in; void main(){}",
                "duplicate layout",
            ), # Finish this expression or collection.
            ("layout(local_size_x=-1) in; void main(){}", "nonnegative integer"), # Apply layout to these arguments.
            ("layout(local_size_x=1) in; void helper(){}", "exactly one void main"), # Apply layout to these arguments.
            ( # Finish this expression or collection.
                "layout(local_size_x=1) in; void main(){} void main(){}",
                "exactly one void main",
            ), # Finish this expression or collection.
            ( # Finish this expression or collection.
                '#include "other.h"\nlayout(local_size_x=1) in; void main(){}',
                "includes are unsupported",
            ), # Finish this expression or collection.
        ]: # Finish this expression or collection.
            with self.subTest(source=source): # Keep this temporary resource alive throughout the following block.
                with self.assertRaisesRegex(COMPILER.ShaderError, message): # Keep this temporary resource alive throughout the following block.
                    COMPILER.generate(source, "invalid_kernel") # Apply generate to these arguments.

    def test_generated_cpp_preserves_native_glsl_semantics(self): # Verify generated cpp preserves native glsl semantics behavior.
        # in parameters may be mutated in GLSL, but the caller's original is unchanged.
        # The block instance and all three native swizzle naming families are exercised.
        source = '''
#ifdef __cplusplus
#include "shdr.h"
#endif
layout(local_size_x=1) in;
layout(std430,binding=0) writeonly buffer Values {float r[];} state;
layout(std430,binding=1) readonly buffer Vectors {vec2 xy[];} vectors;
void adjust(in vec4 value, out vec2 selected, inout uint steps) {
    value.r += 1.0;
    value.yx = value.xy;
    selected = value.st;
    steps += 1u;
}
void main() {
    vec4 original = vec4(1.0,7.0,3.0,4.0);
    vec2 selected = vec2(0.0);
    uint steps = 3u;
    adjust(original,selected,steps);
    state.r[0] = original.r;
    state.r[1] = selected.s;
    state.r[2] = selected.t;
    state.r[3] = float(steps);
    state.r[4] = float(state.r.length());
    float fraction = 0.5;
    fraction = clamp(fraction,0.0,1.0);
    state.r[5] = pow(fraction,2.0) + min(fraction,0.25) + max(fraction,0.5)
                    + mix(0.0,1.0,fraction);
    state.r[6] = 1e-3;
    vec2 pair = vectors.xy[0].xy;
    pair.yx = pair.xy;
    ((state)).r[7] = pair.x;
    ((state)).r[8] = ((vectors)).xy[0].xy.y;
}
'''
        application = '''
#include "shdr.h"
int main() {
    shdr::buffer<float> result(9);
    shdr::buffer<shdr::glsl::vec2> input{shdr::glsl::vec2(2.0f,3.0f)};
    auto shader = shdr::make_kernel("semantics_kernel");
    shader->bind("state.r",result).bind("vectors.xy",input);
    shdr::pipeline pipeline;
    pipeline += shdr::dispatch(*shader,shdr::extent(1));
    pipeline.run();
    const shdr::buffer<float>& read = result;
    const float expected[] = {1.0f,7.0f,2.0f,4.0f,9.0f,1.5f,1e-3f,3.0f,3.0f};
    for(unsigned i=0;i<9;++i) if(read.data()[i]!=expected[i]) return 1;
}
'''
        with tempfile.TemporaryDirectory(prefix="shdr-front-end-") as directory: # Keep this temporary resource alive throughout the following block.
            path = Path(directory) # Compute path for the following operation.
            adapter = path / "adapter.cpp" # Compute adapter for the following operation.
            adapter.write_text(COMPILER.generate(source, "semantics_kernel")) # Apply write text to these arguments.
            main = path / "main.cpp" # Compute main for the following operation.
            main.write_text(application) # Apply write text to these arguments.
            executable = path / "semantics" # Compute executable for the following operation.
            subprocess.run( # Run the compiler, validator or generated executable.
                [ # Finish this expression or collection.
                    CPP_COMPILER, # Supply CPP COMPILER for the continued expression.
                    "-std=c++11",
                    "-pedantic-errors",
                    "-I",
                    str(ROOT), # Apply str to these arguments.
                    str(adapter), # Apply str to these arguments.
                    str(main), # Apply str to these arguments.
                    "-o",
                    str(executable), # Apply str to these arguments.
                ], # Finish this expression or collection.
                check=True, # Compute check for the following operation.
            ) # Finish this expression or collection.
            subprocess.run([str(executable)], check=True, cwd=directory) # Run the compiler, validator or generated executable.

    def test_desktop_numeric_promotions_on_all_profiles(self): # Verify desktop numeric promotions on all profiles behavior.
        source = '''
layout(local_size_x=1) in;
layout(std430,binding=0) buffer Values {float result[];};
layout(std430,binding=1) readonly buffer InputValues {float items[];} input_data;
uniform float runtime_value;
uniform uint unsigned_value;
uniform int signed_value;
const int GLOBAL_COUNT=4;
uint fibonacci(uint n) { if(n<=1) return n; return 2u; }
float scalar_argument(const float input_value) { return input_value*0.5; }
void main() {
    const int LOCAL_COUNT=GLOBAL_COUNT/2;
    float coefficients[LOCAL_COUNT];
    coefficients[0]=1.0;
    coefficients[1]=2.0;
    const float runtime_constant=runtime_value+1.0;
    const uint runtime_count=unsigned_value;
    result[0]=unsigned_value;
    result[1]=runtime_value/signed_value;
    result[2]=1.5/2;
    result[3]=float(fibonacci(unsigned_value));
    result[4]=scalar_argument(4);
    ivec2 integers=mix(ivec2(1,2),ivec2(3,4),bvec2(true,false));
    uvec2 unsigned_integers=mix(uvec2(5,6),uvec2(7,8),bvec2(false,true));
    result[5]=float(integers.x+integers.y);
    result[6]=float(unsigned_integers.x+unsigned_integers.y);
    result[7]=unsigned_value>0 ? runtime_value : 3;
    result[8]=float(unsigned_value==-1);
    result[9]=float(-1<unsigned_value);
    vec2 pair=vec2(runtime_value)/signed_value;
    result[10]=pair.x+runtime_constant+float(runtime_count)+coefficients[1];
    result[11]=mix(0.0,2.0,0.5);
    result[12]=float(unsigned_value<result.length());
    result[13]=float(17u>=result.length());
    result[14]=float(-1<result.length());
    result[15]=float(unsigned_value<input_data.items.length());
    result[16]=float(-1<((input_data)).items.length())+float(coefficients.length());
}
'''
        application = '''
#include "shdr.h"
#include <cmath>
#include <iostream>
int main(int argc,char** argv) {
    auto shader=shdr::make_kernel("promotion_kernel");
    if(argc==2) {
        const std::string name(argv[1]);
        const shdr::profile profile=name=="es31"?shdr::profile::es31:
            name=="es32"?shdr::profile::es32:name=="gl43"?shdr::profile::gl43:
            name=="gl45"?shdr::profile::gl45:shdr::profile::gl46;
        std::cout<<shader->glsl(profile);
        return 0;
    }
    shdr::buffer<float> result(17), input{2.0f,4.0f};
    shader->bind("result",result).bind("input_data.items",input).set<float>("runtime_value",6.0f)
          .set<unsigned>("unsigned_value",3u).set<int>("signed_value",2);
    shdr::pipeline pipeline=shdr::dispatch(*shader,shdr::extent(1));
    pipeline.run();
    const shdr::buffer<float>& read=result;
    const float expected[]={3.0f,3.0f,0.75f,2.0f,2.0f,5.0f,13.0f,6.0f,0.0f,0.0f,15.0f,1.0f,1.0f,1.0f,1.0f,0.0f,3.0f};
    for(unsigned i=0;i<17;++i) if(std::abs(read.data()[i]-expected[i])>1e-6f) return 1;
#ifdef SHDR_ENABLE_EGL
    try {
        shdr::headless_device device;
        pipeline.run(device);
        for(unsigned i=0;i<17;++i) if(std::abs(read.data()[i]-expected[i])>1e-6f) return 2;
    } catch(const shdr::backend_unavailable&) { return 77; }
#endif
    return 0;
}
'''
        _, gpu, _ = COMPILER.transform(source, "promotion_kernel") # Apply transform to these arguments.
        self.assertIn('const int LOCAL_COUNT', gpu) # Check the expected frontend result or execution status.
        self.assertNotIn('const float runtime_constant', gpu) # Check the expected frontend result or execution status.
        self.assertNotIn('const uint runtime_count', gpu) # Check the expected frontend result or execution status.
        with tempfile.TemporaryDirectory(prefix="shdr-promotion-test-") as directory: # Keep this temporary resource alive throughout the following block.
            path = Path(directory) # Compute path for the following operation.
            adapter, main, executable = ( # Supply adapter for the continued expression.
                path / 'adapter.cpp', # Supply path for the continued expression.
                path / 'main.cpp', # Supply path for the continued expression.
                path / 'promotions', # Supply path for the continued expression.
            ) # Finish this expression or collection.
            adapter.write_text(COMPILER.generate(source, "promotion_kernel")) # Apply write text to these arguments.
            main.write_text(application) # Apply write text to these arguments.
            command = [ # Compute command for the following operation.
                CPP_COMPILER, # Supply CPP COMPILER for the continued expression.
                '-std=c++11',
                '-pedantic-errors',
                '-I',
                str(ROOT), # Apply str to these arguments.
                str(adapter), # Apply str to these arguments.
                str(main), # Apply str to these arguments.
                '-o',
                str(executable), # Apply str to these arguments.
            ] # Finish this expression or collection.
            subprocess.run(command, check=True) # Run the compiler, validator or generated executable.
            subprocess.run([str(executable)], check=True) # Run the compiler, validator or generated executable.
            if GLSLANG_VALIDATOR: # Select this branch when its condition holds.
                for profile in ('es31', 'es32', 'gl43', 'gl45', 'gl46'): # Process each profile in this sequence.
                    exported = subprocess.run( # Compute exported for the following operation.
                        [str(executable), profile], # Apply str to these arguments.
                        check=True, # Compute check for the following operation.
                        stdout=subprocess.PIPE, # Compute stdout for the following operation.
                        text=True, # Compute text for the following operation.
                    ).stdout # Supply stdout for the continued expression.
                    shader = path / (profile + '.comp') # Compute shader for the following operation.
                    shader.write_text(exported) # Apply write text to these arguments.
                    subprocess.run( # Run the compiler, validator or generated executable.
                        [GLSLANG_VALIDATOR, '-S', 'comp', str(shader)], check=True # Apply str to these arguments.
                    ) # Finish this expression or collection.
            # A development machine with EGL also checks actual GLES values after translation.
            if EGL_FLAGS: # Select this branch when its condition holds.
                subprocess.run( # Run the compiler, validator or generated executable.
                    command + ['-DSHDR_ENABLE_EGL=1'] + EGL_FLAGS, check=True # Supply command for the continued expression.
                ) # Finish this expression or collection.
                completed = subprocess.run([str(executable)]) # Compute completed for the following operation.
                self.assertIn(completed.returncode, (0, 77)) # Check the expected frontend result or execution status.

    def test_private_global_initializers_are_per_invocation(self): # Recreate private globals while retaining linked resources and snapshotted uniforms.
        source = '''
layout(local_size_x=2) in;
layout(std430,binding=0) writeonly buffer Results {float values[];} state;
layout(r32f,binding=0) readonly uniform highp image2D image_values;
layout(push_constant) uniform Params {float boost;} p;
uniform float gain;
uniform uint count;
#define DECLARE_PRIVATE(name) float name=0.0; float scratch[2]=float[2](2.0,4.0);
DECLARE_PRIVATE(counter)
void update_private() {
    counter+=1.0+p.boost;
    scratch[0]+=gain+imageLoad(image_values,ivec2(0)).x;
    scratch[1]-=2.0;
}
void main() {
    uint index=gl_GlobalInvocationID.x;
    if(index>=count) return;
    update_private();
    state.values[3u*index]=counter;
    state.values[3u*index+1u]=scratch[0]-4.0;
    state.values[3u*index+2u]=scratch[1]-1.0;
}
'''
        uniform_initializer = '''
layout(local_size_x=1) in;
layout(std430,binding=0) writeonly buffer Results {float values[];};
uniform float seed;
float counter=seed;
float scratch[2]=float[2](seed,seed+1.0);
void main() {
    uint index=gl_GlobalInvocationID.x;
    counter+=scratch[0]+scratch[1];
    values[index]=counter;
}
'''
        application = '''
#include "shdr.h"
#include <iostream>
void verify(const shdr::buffer<float>& result,float first,float second,float third) {
    for(unsigned i=0;i<result.size();i+=3u)
        if(result.data()[i]!=first || result.data()[i+1u]!=second || result.data()[i+2u]!=third)
            throw shdr::error("private globals were shared across invocations or dispatches");
}
int main(int argc,char** argv) {
    auto shader=shdr::make_kernel("private_global_kernel");
    if(argc==2) {
        const std::string name(argv[1]);
        const shdr::profile profile=name=="es31"?shdr::profile::es31:name=="es32"?shdr::profile::es32:
            name=="gl43"?shdr::profile::gl43:name=="gl45"?shdr::profile::gl45:shdr::profile::gl46;
        std::cout<<shader->glsl(profile);
        return 0;
    }
    shdr::buffer<float> result(18), other(18);
    shdr::image2d<float,1> image(1,1);
    shader->bind("state.values",result).bind("image_values",image)
          .set<float>("gain",3.0f).set<float>("p.boost",0.0f).set<unsigned>("count",6u);
    shdr::pipeline pipeline=shdr::dispatch(*shader,shdr::extent(3));
    pipeline.run();
    verify(result,1.0f,1.0f,1.0f);
    pipeline.run();
    verify(result,1.0f,1.0f,1.0f);
    shdr::executor worker(1,1);
    auto done=worker.submit(pipeline);
    shader->bind("state.values",other).set<float>("gain",7.0f);
    done.get();
    verify(result,1.0f,1.0f,1.0f);
    verify(other,0.0f,0.0f,0.0f);
    pipeline.run();
    verify(other,1.0f,5.0f,1.0f);
    shader->bind("state.values",result).set<float>("gain",3.0f);
    auto initialized=shdr::make_kernel("bound_initializer_kernel");
    shdr::buffer<float> seeded(4);
    initialized->bind("values",seeded).set<float>("seed",2.0f);
    shdr::pipeline initialized_pipeline=shdr::dispatch(*initialized,shdr::extent(4));
    initialized_pipeline.run();
    const shdr::buffer<float>& read_seeded=seeded;
    for(unsigned i=0;i<4u;++i) if(read_seeded.data()[i]!=7.0f) return 1;
    initialized->set<float>("seed",4.0f);
    initialized_pipeline.run();
    for(unsigned i=0;i<4u;++i) if(read_seeded.data()[i]!=13.0f) return 2;
#ifdef SHDR_ENABLE_EGL
    try {
        shdr::headless_device device;
        pipeline.run(device);
        verify(result,1.0f,1.0f,1.0f);
        pipeline.run(device);
        verify(result,1.0f,1.0f,1.0f);
    } catch(const shdr::backend_unavailable&) { return 77; }
#endif
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="shdr-private-global-test-") as directory: # Keep generated adapters and executions isolated from project files.
            path = Path(directory) # Resolve all regression artifacts within the temporary directory.
            adapter, seeded_adapter, main, executable = path/'adapter.cpp', path/'seeded.cpp', path/'main.cpp', path/'private-globals' # Give each factory its own standard C++ translation unit.
            adapter.write_text(COMPILER.generate(source,"private_global_kernel")) # Compile the same private globals, helper, buffers, image and uniforms used by GPU.
            seeded_adapter.write_text(COMPILER.generate(uniform_initializer,"bound_initializer_kernel")) # Check that global initializer values are read after resource binding.
            main.write_text(application) # Exercise repeated calls and an owned asynchronous kernel snapshot.
            command=[CPP_COMPILER,'-std=c++11','-pedantic-errors','-pthread','-I',str(ROOT),str(adapter),str(seeded_adapter),str(main),'-o',str(executable)] # Compile both adapters with the required baseline language and executor support.
            subprocess.run(command,check=True) # Reject any C++11 portability or frame-reference errors.
            subprocess.run([str(executable)],check=True) # Check every scalar result across invocations, repeat dispatches and a snapshot.
            if GLSLANG_VALIDATOR: # Validate the unchanged GPU variant in every supported language profile.
                for profile in ('es31','es32','gl43','gl45','gl46'): # Cover both GLES versions and every desktop target.
                    exported=subprocess.run([str(executable),profile],check=True,stdout=subprocess.PIPE,text=True).stdout # Export the same source embedded with the registered CPU kernel.
                    shader=path/(profile+'.comp') # Keep each profile's source separate for diagnostics.
                    shader.write_text(exported) # Validate the actual embedded portable shader text.
                    subprocess.run([GLSLANG_VALIDATOR,'-S','comp',str(shader)],check=True) # Require every profile to accept the private-global source.
            if EGL_FLAGS: # A development machine with EGL also executes the actual GLES private-state behavior.
                subprocess.run(command+['-DSHDR_ENABLE_EGL=1']+EGL_FLAGS,check=True) # Enable the optional library backend for the same shader sources.
                completed=subprocess.run([str(executable)]) # Compare the CPU results with two actual GLES dispatches.
                self.assertIn(completed.returncode,(0,77)) # Skip only an unavailable optional context and reject computation failures.
        self.assertTrue(COMPILER.needs_invocation_frame('float first,second=0.0;void main(){}')) # Detect an initialized later declarator in the same global statement.
        self.assertFalse(COMPILER.needs_invocation_frame('const float first=0.0;void main(){}')) # Keep immutable global constants on the byte-identical fast path.
        self.assertTrue(COMPILER.needs_invocation_frame('#define PRIVATE_STATE float counter=0.0;\nPRIVATE_STATE\nvoid main(){}')) # Detect an object macro hiding an initialized global declaration.
        self.assertTrue(COMPILER.needs_invocation_frame('#define PRIVATE_STATE \\\nfloat counter=0.0;\nPRIVATE_STATE\nvoid main(){}')) # Retain assignment detection across a continued macro definition.


    def test_imported_layouts_fixed_arrays_and_helpers(self): # Verify imported layouts fixed arrays and helpers behavior.
        source = '''
#version 430 core
#define BASE (1u << 0u)
#define COUNT 4
const uint INPUT_BINDING = BASE | 0u;
layout(std430) buffer;
layout(local_size_x=BASE) in;
layout(binding=INPUT_BINDING) readonly buffer Inputs {float items[COUNT];} inputs;
layout(binding=BASE+1) writeonly buffer Outputs {float results[COUNT];} outputs;
layout(location=2) uniform float gain;
void weights(out float[COUNT] result) {
    result = float[COUNT](1.0,2.0,3.0,4.0);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if(i >= uint(inputs.items.length())) return;
    float kernel[COUNT];
    weights(kernel);
    const float values[COUNT] = float[COUNT](0.0,1.0,2.0,3.0);
    outputs.results[i] = inputs.items[i]*gain + kernel[i]
                       + sqrt(float(vec2(4.0,9.0))) + float(values.length());
}
'''
        application = '''
#include "shdr.h"
#include <iostream>
int main(int argc, char** argv) {
    auto shader=shdr::make_kernel("import_kernel");
    if(argc==2) {
        const std::string name(argv[1]);
        const shdr::profile profile=name=="es31" ? shdr::profile::es31 :
            name=="es32" ? shdr::profile::es32 : name=="gl43" ? shdr::profile::gl43 :
            name=="gl45" ? shdr::profile::gl45 : shdr::profile::gl46;
        std::cout<<shader->glsl(profile);
        return 0;
    }
    shdr::buffer<float> input{1.f,2.f,3.f,4.f,5.f,6.f}, output(6);
    shader->bind("inputs.items",input).bind("outputs.results",output).set<float>("gain",2.f);
    shdr::pipeline pipeline;
    pipeline+=shdr::dispatch(*shader,shdr::extent(4));
    pipeline.run();
    const shdr::buffer<float>& read=output;
    const float expected[]={9.f,12.f,15.f,18.f,0.f,0.f};
    for(unsigned i=0;i<6;++i) if(read.data()[i]!=expected[i]) return 1;
    shdr::buffer<float> short_input(3);
    shader->bind("inputs.items",short_input);
    try { pipeline.run(); } catch(const shdr::error&) { return 0; }
    return 2;
}
'''
        with tempfile.TemporaryDirectory(prefix="shdr-import-test-") as directory: # Keep this temporary resource alive throughout the following block.
            path = Path(directory) # Compute path for the following operation.
            adapter, main, executable = ( # Supply adapter for the continued expression.
                path / "adapter.cpp", # Supply path for the continued expression.
                path / "main.cpp", # Supply path for the continued expression.
                path / "import", # Supply path for the continued expression.
            ) # Finish this expression or collection.
            adapter.write_text(COMPILER.generate(source, "import_kernel")) # Apply write text to these arguments.
            main.write_text(application) # Apply write text to these arguments.
            subprocess.run( # Run the compiler, validator or generated executable.
                [ # Finish this expression or collection.
                    CPP_COMPILER, # Supply CPP COMPILER for the continued expression.
                    "-std=c++11",
                    "-pedantic-errors",
                    "-I",
                    str(ROOT), # Apply str to these arguments.
                    str(adapter), # Apply str to these arguments.
                    str(main), # Apply str to these arguments.
                    "-o",
                    str(executable), # Apply str to these arguments.
                ], # Finish this expression or collection.
                check=True, # Compute check for the following operation.
            ) # Finish this expression or collection.
            subprocess.run([str(executable)], check=True, cwd=directory) # Run the compiler, validator or generated executable.
            if GLSLANG_VALIDATOR: # Select this branch when its condition holds.
                for profile in ("es31", "es32", "gl43", "gl45", "gl46"): # Process each profile in this sequence.
                    exported = subprocess.run( # Compute exported for the following operation.
                        [str(executable), profile], # Apply str to these arguments.
                        check=True, # Compute check for the following operation.
                        stdout=subprocess.PIPE, # Compute stdout for the following operation.
                        text=True, # Compute text for the following operation.
                    ).stdout # Supply stdout for the continued expression.
                    shader = path / (profile + ".comp") # Compute shader for the following operation.
                    shader.write_text(exported) # Apply write text to these arguments.
                    subprocess.run( # Run the compiler, validator or generated executable.
                        [GLSLANG_VALIDATOR, "-S", "comp", str(shader)], check=True # Apply str to these arguments.
                    ) # Finish this expression or collection.

    def test_imported_descriptor_defaults_and_push_constants(self): # Verify imported descriptor defaults and push constants behavior.
        source = '''
#version 450
layout(constant_id=0) const uint BIAS=3u;
layout(local_size_x_id=2) in;
layout(set=0,binding=1) readonly buffer Input {float values[];} input_data;
writeonly buffer Output {float results[];};
layout(push_constant) uniform Params {float r; uint count;} p;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i<p.count) results[i]=input_data.values[i]*p.r+float(BIAS);
}
'''
        application = '''
#include "shdr.h"
#include <iostream>
int main(int argc,char** argv) {
    auto shader=shdr::make_kernel("descriptor_kernel");
    if(argc==2) {
        const std::string name(argv[1]);
        const shdr::profile p=name=="es31"?shdr::profile::es31:name=="es32"?shdr::profile::es32:
            name=="gl43"?shdr::profile::gl43:name=="gl45"?shdr::profile::gl45:shdr::profile::gl46;
        std::cout<<shader->glsl(p); return 0;
    }
    shdr::buffer<float> input{2.f,5.f},output(2);
    shader->bind("input_data.values",input).bind("results",output)
          .set<float>("p.r",4.f).set<unsigned>("p.count",2u);
    if(shader->local_size().x!=1) return 1;
    shdr::pipeline pipeline=shdr::dispatch(*shader,shdr::extent(2));
    pipeline.run();
    const shdr::buffer<float>& read=output;
    return read.data()[0]==11.f && read.data()[1]==23.f ? 0 : 2;
}
'''
        with tempfile.TemporaryDirectory(prefix="shdr-descriptor-test-") as directory: # Keep this temporary resource alive throughout the following block.
            path = Path(directory) # Compute path for the following operation.
            adapter, main, executable = ( # Supply adapter for the continued expression.
                path / "adapter.cpp", # Supply path for the continued expression.
                path / "main.cpp", # Supply path for the continued expression.
                path / "descriptor", # Supply path for the continued expression.
            ) # Finish this expression or collection.
            adapter.write_text(COMPILER.generate(source, "descriptor_kernel")) # Apply write text to these arguments.
            main.write_text(application) # Apply write text to these arguments.
            subprocess.run( # Run the compiler, validator or generated executable.
                [ # Finish this expression or collection.
                    CPP_COMPILER, # Supply CPP COMPILER for the continued expression.
                    "-std=c++11",
                    "-pedantic-errors",
                    "-I",
                    str(ROOT), # Apply str to these arguments.
                    str(adapter), # Apply str to these arguments.
                    str(main), # Apply str to these arguments.
                    "-o",
                    str(executable), # Apply str to these arguments.
                ], # Finish this expression or collection.
                check=True, # Compute check for the following operation.
            ) # Finish this expression or collection.
            subprocess.run([str(executable)], check=True, cwd=directory) # Run the compiler, validator or generated executable.
            if GLSLANG_VALIDATOR: # Select this branch when its condition holds.
                for profile in ("es31", "es32", "gl43", "gl45", "gl46"): # Process each profile in this sequence.
                    shader = path / (profile + ".comp") # Compute shader for the following operation.
                    exported = subprocess.run( # Compute exported for the following operation.
                        [str(executable), profile], # Apply str to these arguments.
                        check=True, # Compute check for the following operation.
                        stdout=subprocess.PIPE, # Compute stdout for the following operation.
                        text=True, # Compute text for the following operation.
                    ).stdout # Supply stdout for the continued expression.
                    shader.write_text(exported) # Apply write text to these arguments.
                    subprocess.run( # Run the compiler, validator or generated executable.
                        [GLSLANG_VALIDATOR, "-S", "comp", str(shader)], check=True # Apply str to these arguments.
                    ) # Finish this expression or collection.
        _, gpu, local = COMPILER.transform("void main(){}", "default_kernel") # Apply transform to these arguments.
        self.assertEqual(local, [1, 1, 1]) # Check the expected frontend result or execution status.
        self.assertIn("local_size_x=1,local_size_y=1,local_size_z=1", gpu) # Check the expected frontend result or execution status.

    def test_helper_constant_shadow_does_not_change_storage_layout(self): # Verify storage layout uses the enclosing constant scope.
        source = '''
const uint WIDTH=4u;
layout(local_size_x=WIDTH) in;
layout(std430,binding=0) writeonly buffer Output {float values[WIDTH];};
float helper(){
    const uint WIDTH=2u;
    float items[WIDTH];
    const vec2 first[WIDTH]=vec2[WIDTH](vec2(1.0,2.0),vec2(3.0,4.0)),
               second[WIDTH]=vec2[WIDTH](vec2(5.0,6.0),vec2(7.0,8.0));
    float a[WIDTH]=float[WIDTH](1.0,2.0), scalar=3.0, b[WIDTH]=float[WIDTH](4.0,5.0);
    float result=first[1].y+second[0].x+float(items.length())+a[1]+b[0]+scalar;
    for(const uint WIDTH=3u;WIDTH==3u;){float temporary[WIDTH];result+=float(temporary.length());break;}
    float after_loop[WIDTH];
    return result+float(after_loop.length());
}
void main(){
    for(const uint WIDTH=2u;WIDTH==2u;){float temporary[WIDTH];if(temporary.length()!=2)return;break;}
    float after_loop[WIDTH];
    values[gl_GlobalInvocationID.x]=float(values.length())+helper()+float(after_loop.length());
}
'''
        _, _, local = COMPILER.transform(source, "shadow_kernel") # Apply transform to these arguments.
        self.assertEqual(local, [4, 1, 1]) # Check the expected frontend result or execution status.
        application = '''
#include "shdr.h"
int main() {
    shdr::buffer<float> output(4);
    auto shader=shdr::make_kernel("shadow_kernel");
    shader->bind("values",output);
    shdr::pipeline pipeline=shdr::dispatch(*shader,shdr::extent(1));
    pipeline.run();
    const shdr::buffer<float>& read=output;
    for(unsigned i=0;i<4;++i) if(read.data()[i]!=33.f) return 1;
}
'''
        with tempfile.TemporaryDirectory(prefix="shdr-constant-scope-") as directory: # Keep this temporary resource alive throughout the following block.
            path = Path(directory) # Compute path for the following operation.
            adapter, main, executable = ( # Supply adapter for the continued expression.
                path / "adapter.cpp", # Supply path for the continued expression.
                path / "main.cpp", # Supply path for the continued expression.
                path / "shadow", # Supply path for the continued expression.
            ) # Finish this expression or collection.
            adapter.write_text(COMPILER.generate(source, "shadow_kernel")) # Apply write text to these arguments.
            main.write_text(application) # Apply write text to these arguments.
            subprocess.run( # Run the compiler, validator or generated executable.
                [ # Finish this expression or collection.
                    CPP_COMPILER, # Supply CPP COMPILER for the continued expression.
                    "-std=c++11",
                    "-pedantic-errors",
                    "-I",
                    str(ROOT), # Apply str to these arguments.
                    str(adapter), # Apply str to these arguments.
                    str(main), # Apply str to these arguments.
                    "-o",
                    str(executable), # Apply str to these arguments.
                ], # Finish this expression or collection.
                check=True, # Compute check for the following operation.
            ) # Finish this expression or collection.
            subprocess.run([str(executable)], check=True, cwd=directory) # Run the compiler, validator or generated executable.


if __name__ == "__main__": # Select this branch when its condition holds.
    arguments = argparse.ArgumentParser(description=__doc__) # Compute arguments for the following operation.
    arguments.add_argument( # Apply add argument to these arguments.
        "--cxx", required=True, help="C++11 compiler used by the library build" # Supply cxx for the continued expression.
    ) # Finish this expression or collection.
    arguments.add_argument( # Apply add argument to these arguments.
        "--glslang", help="optional GLSL validator for all supported import profiles" # Supply glslang for the continued expression.
    ) # Finish this expression or collection.
    arguments.add_argument( # Apply add argument to these arguments.
        "--egl",
        action="store_true", # Compute action for the following operation.
        help="check portable values on an optional headless EGL backend", # Compute help for the following operation.
    ) # Finish this expression or collection.
    options, unittest_arguments = arguments.parse_known_args() # Apply parse known args to these arguments.
    if options.egl: # Select this branch when its condition holds.
        import shlex # Import shlex for this operation.

        EGL_FLAGS = shlex.split( # Compute EGL FLAGS for the following operation.
            subprocess.run( # Run the compiler, validator or generated executable.
                ["pkg-config", "--cflags", "--libs", "egl"], # Supply pkg for the continued expression.
                check=True, # Compute check for the following operation.
                stdout=subprocess.PIPE, # Compute stdout for the following operation.
                text=True, # Compute text for the following operation.
            ).stdout # Supply stdout for the continued expression.
        ) # Finish this expression or collection.
    CPP_COMPILER = options.cxx # Compute CPP COMPILER for the following operation.
    GLSLANG_VALIDATOR = options.glslang # Compute GLSLANG VALIDATOR for the following operation.
    unittest.main(argv=[__file__] + unittest_arguments) # Apply main to these arguments.
