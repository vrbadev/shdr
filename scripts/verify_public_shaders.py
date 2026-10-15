#!/usr/bin/env python3
"""Compile pinned public compute shaders and record their compiler outcomes.

Upstream sources reside in a separate cache. Reference checks use standalone
inputs or recorded upstream configurations. CPU checks compile generated C++11
adapters, and accepted adapters export GPU programs for all five profiles.
Numerical checks require application resources, valid inputs and reference values.
"""

import argparse  # Read command-line options and show help.
from collections import Counter  # Count each compiler outcome by category.
from concurrent.futures import ThreadPoolExecutor  # Check independent sources with a bounded set of workers.
import hashlib  # Identify exact source and header snapshots.
import json  # Write compiler results as a machine-readable report.
from pathlib import Path  # Build paths and read or write files.
import re  # Find declarations, comments and diagnostic lines.
import subprocess  # Run compilers with direct argument lists.
import tempfile  # Keep intermediate source and binaries in temporary directories.
import types  # Load the frontend into an isolated module object.
from report_public_shaders import render_public_report  # Render complete compiler evidence as a standalone HTML document.


ROOT = Path(__file__).resolve().parents[1]  # Find the checkout containing the maintained source files.
PROFILES = ("es31", "es32", "gl43", "gl45", "gl46")  # List every supported shader-language profile.
REPOSITORIES = (  # Pin upstream revisions and record their shader locations.
    {  # Describe this pinned upstream source collection.
        "id": "glslang",  # Name the repository and its cache folder.
        "url": "https://github.com/KhronosGroup/glslang.git",  # Record the upstream Git URL.
        "revision": "2fbed5e1fb43a1c480fe020e3bb686a31010989a",  # Pin the exact upstream commit.
        "paths": ["Test"],  # Select the tracked directories containing shader inputs.
        "license_url": "https://github.com/KhronosGroup/glslang/blob/2fbed5e1fb43a1c480fe020e3bb686a31010989a/LICENSE.txt",  # Link the upstream license for reuse decisions.
        "description": "Khronos reference compiler compute fixtures, including negative tests",  # Describe this source collection.
    },  # Complete the pinned Khronos compiler fixture collection.
    {  # Describe this pinned upstream source collection.
        "id": "ogl-samples",  # Name the repository and its cache folder.
        "url": "https://github.com/g-truc/ogl-samples.git",  # Record the upstream Git URL.
        "revision": "38cada7a9458864265e25415ae61586d500ff5fc",  # Pin the exact upstream commit.
        "paths": ["data"],  # Select the tracked directories containing shader inputs.
        "license_url": "https://github.com/g-truc/ogl-samples/blob/38cada7a9458864265e25415ae61586d500ff5fc/LICENSE.md",  # Link the upstream license for reuse decisions.
        "description": "MIT-licensed OpenGL sample compute shaders",  # Describe this source collection.
    },  # Complete the pinned desktop OpenGL sample collection.
    {  # Describe this pinned upstream source collection.
        "id": "vulkan",  # Name the repository and its cache folder.
        "url": "https://github.com/SaschaWillems/Vulkan.git",  # Record the upstream Git URL.
        "revision": "4ee2501c8280cf71d1f9ce231cd8ebd47b58adde",  # Pin the exact upstream commit.
        "paths": ["shaders/glsl"],  # Select the tracked directories containing shader inputs.
        "license_url": "https://github.com/SaschaWillems/Vulkan/blob/4ee2501c8280cf71d1f9ce231cd8ebd47b58adde/LICENSE.md",  # Link the upstream license for reuse decisions.
        "description": "MIT-licensed Vulkan example GLSL compute shaders",  # Describe this source collection.
    },  # Complete the pinned Vulkan example collection.
    {  # Describe this pinned upstream source collection.
        "id": "kompute",  # Name the repository and its cache folder.
        "url": "https://github.com/KomputeProject/kompute.git",  # Record the upstream Git URL.
        "revision": "267e019ec419ae57f634b341983b60972924b58f",  # Pin the exact upstream commit.
        "paths": ["examples", "test", "benchmark"],  # Select the tracked directories containing shader inputs.
        "license_url": "https://github.com/KomputeProject/kompute/blob/267e019ec419ae57f634b341983b60972924b58f/LICENSE",  # Link the upstream license for reuse decisions.
        "description": "Apache-2.0-licensed general-purpose compute examples",  # Describe this source collection.
    },  # Complete the pinned general compute collection.
    {  # Describe this pinned upstream source collection.
        "id": "ncnn",  # Name the repository and its cache folder.
        "url": "https://github.com/Tencent/ncnn.git",  # Record the upstream Git URL.
        "revision": "fcde729c7b0b0fcecf273ed21b0e25dd2deb67be",  # Pin the exact upstream commit.
        "paths": ["src/layer/vulkan/shader"],  # Select the tracked directories containing shader inputs.
        "license_url": "https://github.com/Tencent/ncnn/blob/fcde729c7b0b0fcecf273ed21b0e25dd2deb67be/LICENSE.txt",  # Link the upstream license for reuse decisions.
        "description": "BSD-3-Clause-licensed neural-network compute kernels with the upstream FP32 buffer prelude",  # Describe this source collection.
        "configuration": "ncnn_fp32_buffer",  # Record any required upstream build configuration.
        "prelude_path": "src/ncnn_glsl_ext.comp",  # Locate the upstream macro and type prelude.
        "configuration_source_url": "https://github.com/Tencent/ncnn/blob/fcde729c7b0b0fcecf273ed21b0e25dd2deb67be/src/gpu.cpp#L6285",  # Link the upstream source-assembly implementation.
    },  # Complete the pinned configured neural-network collection.
)  # Keep all upstream metadata beside the selected revisions.


def digest(data):  # Identify exact source bytes with SHA-256.
    return hashlib.sha256(data).hexdigest()  # Return the hexadecimal source fingerprint.


def run(command, cwd=None, timeout=45, input_text=None):  # Capture a compiler exit status and its diagnostic output.
    """Run compiler tools through direct argument lists."""
    try:  # Catch a compiler that exceeds its per-case timeout.
        result = subprocess.run(  # Capture this command's status and output.
            command,  # Pass each compiler argument directly.
            cwd=cwd,  # Run relative paths in the requested working directory.
            stdout=subprocess.PIPE,  # Capture standard output for the report.
            stderr=subprocess.STDOUT,  # Keep compiler errors in the same captured output.
            text=True,  # Decode process output as text.
            timeout=timeout,  # Stop a compiler that exceeds this case's time limit.
            input=input_text,  # Supply embedded GLSL text to the exporter through standard input.
        )  # Capture the combined compiler output when the process finishes.
        return result.returncode, result.stdout  # Return the compiler status and captured diagnostic text.
    except subprocess.TimeoutExpired:  # Treat an expired compiler as a failed check with a distinct status.
        return 124, "compiler timeout"  # Return timeout status 124 with a short explanation.


def checked(command, cwd=None):  # Require a helper command to finish successfully.
    code, output = run(command, cwd=cwd, timeout=300)  # Allow cache and helper commands a longer timeout.
    if code:  # Handle a nonzero compiler status as a failed check.
        raise RuntimeError(output.strip())  # Report helper failures before relying on their output.
    return output.strip()  # Return the successful command output.


def prepare_cache(cache, fetch):  # Check each pinned repository before reading its sources.
    """Read pinned snapshots. Download them only when --fetch is supplied."""
    cache.mkdir(parents=True, exist_ok=True)  # Create the cache parent and preserve existing snapshots.
    for repository in REPOSITORIES:  # Visit every pinned public source collection.
        folder = cache / repository["id"]  # Locate the cached upstream repository.
        if not (folder / ".git").exists():  # Create a Git snapshot for a fresh cache.
            if not fetch:  # Require explicit download permission through the fetch option.
                raise RuntimeError("missing " + repository["id"] + " cache. Run with --fetch first")  # Explain how to populate a missing pinned cache.
            if folder.exists() and any(folder.iterdir()):  # Avoid placing a repository into an unrelated nonempty directory.
                raise RuntimeError("cache destination is not an empty repository: " + str(folder))  # Protect unrelated files in a nonempty cache destination.
            folder.mkdir(parents=True, exist_ok=True)  # Create the empty cache directory before Git uses it.
            checked(["git", "init", "--quiet", str(folder)])  # Initialize an empty repository for this missing cache.
            checked(["git", "remote", "add", "origin", repository["url"]], cwd=folder)  # Record the upstream URL as this repository's origin.
            checked(  # Fetch the pinned commit with bounded download history.
                [  # List the Git arguments for the pinned source download.
                    "git",  # Use Git for the pinned source download.
                    "fetch",  # Download the chosen revision before sparse checkout.
                    "--depth=1",  # Avoid downloading unrelated commit history.
                    "--filter=blob:none",  # Delay file downloads until the selected paths are checked out.
                    "origin",  # Read from the recorded upstream repository.
                    repository["revision"],  # Fetch the exact pinned commit.
                ],  # Complete the arguments for downloading only the pinned revision.
                cwd=folder,  # Run this Git operation inside its cache repository.
            )  # Fetch inside this upstream cache repository.
            checked(["git", "sparse-checkout", "init", "--cone"], cwd=folder)  # Enable sparse checkout to limit cached source files.
            checked(["git", "sparse-checkout", "set"] + repository["paths"], cwd=folder)  # Select only the repository paths listed for this collection.
            checked(["git", "checkout", "--quiet", "--detach", repository["revision"]], cwd=folder)  # Check out the pinned commit in detached mode.
        revision = checked(["git", "rev-parse", "HEAD"], cwd=folder)  # Read the revision currently checked out in the cache.
        if revision != repository["revision"]:  # Require the cache to match its pinned commit.
            raise RuntimeError(  # Require the recorded snapshot revision.
                repository["id"] + " cache is at a different revision. Use a fresh cache directory"  # Identify the cache revision validation error.
            )  # Name the mismatched cache in the failure message.
        if checked(["git", "status", "--porcelain", "--untracked-files=no"], cwd=folder):  # Reject modified tracked inputs before compiler checks begin.
            raise RuntimeError(repository["id"] + " cache contains modified tracked files")  # Reject tracked edits that would change the recorded upstream inputs.


def cases_from_cache(cache):  # Select tracked compute sources and record negative-test metadata.
    cases = []  # Collect the selected source records.
    excluded_hlsl = 0  # Count separately identified HLSL files.
    for repository in REPOSITORIES:  # Visit every pinned public source collection.
        folder = cache / repository["id"]  # Locate the cached upstream repository.
        tracked = checked(["git", "ls-files", "-z", "--"] + repository["paths"], cwd=folder)  # Read the repository's tracked source paths.
        for relative in sorted(tracked.split("\0")):  # Visit each tracked shader path in stable order.
            if not relative.endswith(".comp"):  # Select compute GLSL files and skip unrelated tracked files.
                continue  # Skip this item and inspect the next one.
            source = folder / relative  # Locate the tracked compute source inside this repository.
            if "hlsl." in source.name:  # Select filenames belonging to the GLSL collection.
                excluded_hlsl += 1  # Count separately identified HLSL files.
                continue  # Move to the next path after counting an HLSL input.
            golden = folder / "Test" / "baseResults" / (source.name + ".out")  # Locate any upstream reference transcript for this fixture.
            reference_has_errors = golden.exists() and "ERROR:" in golden.read_text(  # Remember whether upstream marked this fixture as an error case.
                errors="replace"  # Read upstream transcripts even when they contain unusual bytes.
            )  # Finish reading the optional upstream reference transcript.
            cases.append((repository, relative, reference_has_errors))  # Keep this tracked source and its upstream test marker.
    return cases, excluded_hlsl  # Return selected GLSL cases and the separate HLSL count.


def diagnostic(text, substitutions):  # Keep a concise compiler explanation with generic paths.
    """Keep concise compiler reasons with generic cache paths."""
    for original, replacement in substitutions:  # Remove each machine-specific path from the diagnostic.
        text = text.replace(str(original), replacement)  # Replace private paths while keeping the compiler explanation.
    lines = [  # Collect only the most useful diagnostic lines.
        line.strip()  # Remove diagnostic indentation before storing it.
        for line in text.splitlines()  # Inspect each compiler output line for useful errors.
        if "error" in line.lower() or "ERROR" in line or "unsupported" in line.lower()  # Keep the compiler's substantive error messages.
    ]  # Keep the selected error lines in their original order.
    return "\n".join(lines[:4])[:1200] if lines else text.strip()[:500]  # Return a short portable explanation.


def reason_category(reason):  # Group rejected constructs by their reported reason.
    for needle, category in (  # Classify reported resource and language requirements.
        ("shared memory", "workgroup_synchronization_or_atomics"),  # Group workgroup-state and synchronization failures.
        ("uniform type", "uniform_or_sampler_interface"),  # Group uniform and sampler interface diagnostics.
        ("buffer blocks", "storage_buffer_layout"),  # Group storage-block declaration diagnostics.
        ("storage buffers", "storage_buffer_layout"),  # Group incompatible storage element layouts.
        ("image", "image_format_or_interface"),  # Group image format and interface diagnostics.
        ("layout", "layout_or_execution_qualifier"),  # Group resource and execution qualifier diagnostics.
        ("include", "external_shader_dependency"),  # Group unresolved source dependencies.
        ("main", "entrypoint_or_linked_unit"),  # Group missing standalone entry points.
        ("#version", "source_version_directive"),  # Group source-version problems.
    ):  # Test these categories in order so the first match is stable.
        if needle in reason:  # Use the first category whose keyword appears in the rejection reason.
            return category  # Return the matching rejection category.
    return "frontend_body_or_declaration"  # Return the matching rejection category.


def configured_source(repository, source, data, cache):  # Assemble the documented NCNN FP32 buffer configuration.
    """Assemble NCNN's documented FP32-buffer source and preserve its upstream body."""
    if repository.get("configuration") != "ncnn_fp32_buffer":  # Use the upstream configuration only when this source needs it.
        return data.decode("utf-8"), None  # Return the configured source and its provenance.
    header = cache / repository["id"] / repository["prelude_path"]  # Locate the upstream source prelude.
    prelude = header.read_bytes()  # Read the exact prelude bytes for configuration provenance.
    defines = {  # Select the documented FP32 buffer configuration.
        "NCNN_fp16_storage": 0,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_fp16_packed": 0,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_fp16_arithmetic": 0,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_bf16_storage": 0,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_bf16_packed": 0,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_image_storage": 0,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_shader_local_memory": 1,  # Set this upstream feature flag for FP32 buffer shaders.
        "NCNN_moltenvk": 0,  # Set this upstream feature flag for FP32 buffer shaders.
    }  # Finish the upstream configuration description.
    options = "#extension GL_GOOGLE_include_directive : require\n"  # Enable the include directive required by the upstream NCNN configuration.
    options += (  # Add all selected feature definitions before the upstream prelude.
        "\n".join("#define {} {}".format(name, value) for name, value in defines.items()) + "\n"  # Emit the selected upstream feature definitions before its shader body.
    )  # Finish the newline-terminated upstream configuration flags.
    text = data.decode("utf-8")  # Assemble the shader text used by this compiler check.
    version = re.search(r"(?m)^[ \t]*#\s*version[^\n]*(?:\n|$)", text)  # Locate the input version directive before inserting its prelude.
    if version:  # Insert configuration text after the existing version directive.
        text = (  # Assemble the shader text used by this compiler check.
            text[: version.end()] + options + prelude.decode("utf-8") + "\n" + text[version.end() :]  # Keep the upstream body unchanged while inserting its configuration.
        )  # Keep the existing version first, followed by configuration and shader body.
    else:  # Supply a desktop version for an unversioned upstream body.
        text = "#version 450\n" + options + prelude.decode("utf-8") + "\n" + text  # Assemble the shader text used by this compiler check.
    metadata = {  # Record how this configured source was assembled.
        "name": repository["configuration"],  # Name the configuration used for this source.
        "defines": defines,  # Record the selected upstream macro values.
        "prelude_path": repository["prelude_path"],  # Locate the upstream macro and type prelude.
        "prelude_sha256": digest(prelude),  # Identify the exact prelude bytes.
        "include_directory": str(source.parent.relative_to(cache / repository["id"])),  # Record the repository-relative include directory.
        "source_assembly_url": repository["configuration_source_url"],  # Link the source of the configuration recipe.
    }  # Finish the upstream configuration description.
    return text, metadata  # Return the configured source and its provenance.


def verify_case(case, cache, options, frontend):  # Check one source through the reference, CPU and GPU compilers.
    repository, relative, reference_has_errors = case  # Read the selected source and its upstream error marker.
    source = cache / repository["id"] / relative  # Locate the original upstream source file.
    data = source.read_bytes()  # Read the original upstream shader bytes.
    record = {  # Start the report entry for this exact source.
        "repository": repository["id"],  # Identify the upstream project for this source.
        "path": relative,  # Record the source path relative to its repository.
        "source_url": repository["url"][:-4] + "/blob/" + repository["revision"] + "/" + relative,  # Link the exact pinned upstream source.
        "sha256": digest(data),  # Identify the original source bytes.
        "upstream_reference_has_errors": reference_has_errors,  # Keep the upstream negative-test marker.
    }  # Finish recording the original source and its upstream error marker.
    substitutions = [(cache, "<corpus>"), (ROOT, "<shdr>")]  # Replace temporary and checkout paths in diagnostics.
    with tempfile.TemporaryDirectory(prefix="shdr-public-check-") as directory:  # Remove these intermediate compiler files when the check ends.
        generated = Path(directory)  # Locate this case's temporary compiler files.
        substitutions.append((generated, "<generated>"))  # Remove this temporary directory from compiler diagnostics.
        text, configuration = configured_source(repository, source, data, cache)  # Assemble the shader text used by this compiler check.
        reference = source  # Choose the standalone source supplied to the reference compiler.
        include = []  # Pass upstream include paths only when its configuration needs them.
        if configuration:  # Use the upstream configuration only when this source needs it.
            record["configuration"] = configuration  # Record the upstream configuration used to assemble this source.
            reference = generated / "configured.comp"  # Choose the standalone source supplied to the reference compiler.
            reference.write_text(text)  # Write the configured source used by the reference compiler.
            record["configured_sha256"] = digest(text.encode())  # Fingerprint the configured source before any frontend processing.
            include = ["-I" + str(source.parent)]  # Pass upstream include paths only when its configuration needs them.
        opengl = [options.glslang, "-S", "comp", "-l"] + include + [str(reference)]  # Build the reference OpenGL compile-and-link command.
        code, output = run(opengl)  # Compile and link the reference source in OpenGL mode.
        record["opengl"] = {"status": "passed" if code == 0 else "rejected", "exit_code": code}  # Keep this compiler status or diagnostic with the source record.
        if code:  # Handle a nonzero compiler status as a failed check.
            record["opengl"]["diagnostic"] = diagnostic(output, substitutions)  # Keep this compiler status or diagnostic with the source record.
            vulkan = (  # Build the reference Vulkan SPIR-V fallback command.
                [options.glslang, "-V", "-S", "comp", "-l", "-o", str(generated / "shader.spv")]  # Select Vulkan compute compilation and a temporary SPIR-V output.
                + include  # Keep any required upstream include directory.
                + [str(reference)]  # Complete the Vulkan fallback command with the selected source path.
            )  # Finish this compiler command or source-result record.
            vk_code, vk_output = run(vulkan)  # Try the Vulkan reference path when OpenGL mode rejects the input.
            record["vulkan"] = {  # Keep this compiler status or diagnostic with the source record.
                "status": "passed" if vk_code == 0 else "rejected",  # Record whether this compiler accepted the source.
                "exit_code": vk_code,  # Record the Vulkan fallback status alongside the OpenGL result.
            }  # Finish this compiler command or source-result record.
            if vk_code:  # Handle a nonzero compiler status as a failed check.
                record["vulkan"]["diagnostic"] = diagnostic(vk_output, substitutions)  # Keep this compiler status or diagnostic with the source record.
        else:  # Avoid a redundant Vulkan check after OpenGL compilation succeeds.
            record["vulkan"] = {"status": "not_needed"}  # Keep this compiler status or diagnostic with the source record.

        if code and record["vulkan"]["status"] != "passed":  # Stop adaptation when neither reference compiler accepted this source.
            record["cpu"] = {  # Keep this compiler status or diagnostic with the source record.
                "status": "not_attempted",  # Record whether this compiler accepted the source.
                "reason": "reference_compiler_did_not_accept_standalone_source",  # Explain the prerequisite for this compiler check.
            }  # Record the CPU check's reference-acceptance prerequisite.
            return record  # Return the completed compiler record for this source.
        name = "public_" + digest(relative.encode())[:16] + "_kernel"  # Make a unique valid C++ factory name from the source path.
        if configuration:  # Use the upstream configuration only when this source needs it.
            preprocess = [options.glslang, "-E", "-S", "comp"] + include + [str(reference)]  # Ask glslang to resolve the recorded upstream configuration.
            preprocessed_code, text = run(preprocess)  # Resolve upstream macros and includes before CPU translation.
            if preprocessed_code:  # Handle a nonzero compiler status as a failed check.
                record["cpu"] = {  # Keep this compiler status or diagnostic with the source record.
                    "status": "frontend_rejected",  # Record whether this compiler accepted the source.
                    "category": "upstream_configuration_preprocessing",  # Classify the rejected source construct.
                    "diagnostic": diagnostic(text, substitutions),  # Record the upstream preprocessing failure before CPU translation.
                }  # Finish this compiler command or source-result record.
                return record  # Return the preprocessing failure as this source's result.
            record["preprocessed_sha256"] = digest(text.encode())  # Fingerprint the upstream-preprocessed text supplied to the frontend.
        try:  # Catch translation errors while keeping this source's compiler record.
            adapter = frontend.generate(text, name)  # Translate supported GLSL into standard C++11.
        except (frontend.ShaderError, UnicodeError) as error:  # Record the frontend syntax diagnostic as a source rejection.
            reason = str(error)  # Keep the frontend's rejection explanation.
            record["cpu"] = {  # Keep this compiler status or diagnostic with the source record.
                "status": "frontend_rejected",  # Record whether this compiler accepted the source.
                "category": reason_category(reason),  # Classify the rejected source construct.
                "diagnostic": reason,  # Keep the short rejection explanation.
            }  # Finish recording the frontend rejection category and explanation.
            return record  # Return the completed compiler record for this source.
        adapter_path = generated / "adapter.cpp"  # Locate the generated CPU translation unit.
        adapter_path.write_text(adapter)  # Write the generated CPU translation unit.
        command = [  # Build the strict C++11 compiler invocation.
            options.cxx,  # Use the requested C++ compiler.
            "-std=c++11",  # Require the library's minimum C++ language standard.
            "-pedantic-errors",  # Treat nonstandard C++ syntax as an error.
            "-I",  # Add the runtime header directory to the include search path.
            str(ROOT),  # Read the runtime header from this checkout.
            "-fsyntax-only",  # Compile the generated CPU adapter to an object file.
            str(adapter_path),  # Compile this source's generated CPU translation unit.
        ]  # Finish this compiler command or source-result record.
        cpu_code, cpu_output = run(command)  # Complete the strict syntax-only C++11 compiler command.
        record["cpu"] = {  # Keep this compiler status or diagnostic with the source record.
            "status": "compiled" if cpu_code == 0 else "cpp_rejected",  # Record whether this compiler accepted the source.
            "exit_code": cpu_code,  # Record the compiler's numeric exit status.
        }  # Finish this compiler command or source-result record.
        if cpu_code:  # Record the C++ compiler status for the generated adapter.
            record["cpu"]["diagnostic"] = diagnostic(cpu_output, substitutions)  # Keep this compiler status or diagnostic with the source record.
        else:  # Check GPU profiles only after the CPU adapter compiled.
            _, gpu, _ = frontend.transform(text, name)  # Read the GPU declarations and body from the same translated source.
            record["generated_gpu"] = {}  # Create the per-profile result map for this accepted adapter.
            for profile in PROFILES:  # Check every supported shader-language profile.
                export_code, exported = run([str(options.exporter), profile], input_text=gpu)  # Export the embedded source with this profile's runtime preamble.
                if export_code:  # Handle a nonzero compiler status as a failed check.
                    raise RuntimeError(  # Classify exporter failures as infrastructure errors.
                        "embedded GLSL export failed: " + diagnostic(exported, substitutions)  # Include the export failure with generic temporary paths.
                    )  # Report the exporter failure with portable diagnostic paths.
                shader = generated / (profile + ".comp")  # Locate the exported source for this GPU profile.
                shader.write_text(exported)  # Write the runtime-exported GLSL for this profile.
                gpu_code, gpu_output = run([options.glslang, "-S", "comp", "-l", str(shader)])  # Compile and link the exported compute source for this profile.
                result = {  # Capture this command's status and output.
                    "status": "passed" if gpu_code == 0 else "rejected",  # Record whether this compiler accepted the source.
                    "exit_code": gpu_code,  # Record the compiler's numeric exit status.
                }  # Finish the shader-language result for this GPU profile.
                if gpu_code:  # Handle a nonzero compiler status as a failed check.
                    result["diagnostic"] = diagnostic(gpu_output, substitutions)  # Store the profile-specific language error when compilation fails.
                record["generated_gpu"][profile] = result  # Attach this compiler result to its GPU profile name.
    return record  # Return the completed compiler record for this source.


def summary(records):  # Count accepted and rejected sources for the report.
    counts = Counter()  # Count outcomes across the complete collection.
    repositories = {}  # Keep a separate tally for each upstream project.
    reasons = Counter()  # Count frontend rejection categories.
    for record in records:  # Count each source's compiler outcomes.
        count = repositories.setdefault(record["repository"], Counter())  # Select this repository's outcome counters.
        for tally in (counts, count):  # Update the collection total and this repository total.
            tally["sources"] += 1  # Count this compiler outcome for the current total.
            if record["upstream_reference_has_errors"]:  # Count the upstream fixtures intentionally associated with errors.
                tally["upstream_reference_has_errors"] += 1  # Count this compiler outcome for the current total.
            if record["opengl"]["status"] == "passed":  # Count original sources accepted in OpenGL mode.
                tally["opengl_passed"] += 1  # Count this compiler outcome for the current total.
                if not record["upstream_reference_has_errors"]:  # Separate accepted fixtures from upstream negative-test cases.
                    tally["opengl_passed_without_reference_errors"] += 1  # Count this compiler outcome for the current total.
                if record["cpu"]["status"] == "compiled":  # Count generated adapters accepted by standard C++11.
                    tally["opengl_cpu_compiled"] += 1  # Count this compiler outcome for the current total.
            elif record["vulkan"]["status"] == "passed":  # Handle this source form.
                tally["vulkan_only_passed"] += 1  # Count this compiler outcome for the current total.
            else:  # Count sources rejected by both reference compilation paths.
                tally["reference_rejected"] += 1  # Count this compiler outcome for the current total.
            tally["cpu_" + record["cpu"]["status"]] += 1  # Count this compiler outcome for the current total.
            if record["cpu"]["status"] == "compiled":  # Count generated adapters accepted by standard C++11.
                for profile, result in record["generated_gpu"].items():  # Count successful GPU profile checks.
                    if result["status"] == "passed":  # Count this generated GPU profile only when its compiler passed.
                        tally["cpu_and_" + profile + "_compiled"] += 1  # Count this compiler outcome for the current total.
                if all(result["status"] == "passed" for result in record["generated_gpu"].values()):  # Count universal language coverage only when all five profiles passed.
                    tally["cpu_and_all_profiles_compiled"] += 1  # Count this compiler outcome for the current total.
        if record["cpu"]["status"] == "frontend_rejected":  # Count the reason when the adapter frontend rejects a construct.
            reasons[record["cpu"]["category"]] += 1  # Count this frontend rejection category.
    return {  # Return the collection, repository and rejection totals.
        "counts": dict(counts),  # Store totals across all selected sources.
        "repositories": {name: dict(count) for name, count in repositories.items()},  # Record the pinned source collections.
        "frontend_rejection_categories": dict(reasons),  # Store the resource and language rejection categories.
    }  # Return all counters together with the original source records.


def main():  # Read options and run this command-line tool.
    parser = argparse.ArgumentParser(description=__doc__)  # Describe the command and its supported options.
    parser.add_argument("--cache", type=Path, default=Path("build/public-shaders"))  # Select the cache containing the pinned upstream repositories.
    parser.add_argument(  # Offer an explicit option for downloading missing pinned snapshots.
        "--fetch", action="store_true", help="download only the pinned public repository snapshots"  # Describe the download option in the command help.
    )  # Keep downloads disabled unless the fetch flag is supplied.
    parser.add_argument("--glslang", default="glslangValidator")  # Choose the reference GLSL compiler executable.
    parser.add_argument("--cxx", default="c++")  # Choose the compiler used for strict C++11 adapter checks.
    parser.add_argument("--jobs", type=int, default=4)  # Bound the number of concurrent source checks.
    parser.add_argument("--output", type=Path, default=Path("build/public-shader-results.html"))  # Choose where to write the complete browser-readable report.
    options = parser.parse_args()  # Read compiler, cache and output settings.
    if options.jobs < 1:  # Require a positive number of compiler workers.
        parser.error("--jobs must be positive")  # Stop with a helpful command-line usage error.
    cache = options.cache.resolve()  # Resolve the public source cache path.
    try:  # Capture failures from the following compiler or file operations.
        prepare_cache(cache, options.fetch)  # Verify cache revisions and download only when explicitly requested.
        cases, excluded_hlsl = cases_from_cache(cache)  # Collect the selected source records.
        frontend_bytes = (ROOT / "scripts" / "compile_shader.py").read_bytes()  # Capture the frontend snapshot before checks begin.
        header_bytes = (ROOT / "shdr.h").read_bytes()  # Capture the runtime-header snapshot before checks begin.
        frontend = types.ModuleType("shdr_public_frontend")  # Create an isolated namespace for the captured frontend.
        exec(compile(frontend_bytes, "compile_shader.py", "exec"), frontend.__dict__)  # Load only the captured local frontend code into this module.
        with tempfile.TemporaryDirectory(prefix="shdr-public-export-") as export_directory:  # Remove these intermediate compiler files when the check ends.
            exporter_source = Path(export_directory) / "export.cpp"  # Locate the temporary GLSL exporter source.
            exporter_source.write_text(  # Write the small helper that exports this runtime's GLSL preamble.
                """#include "shdr.h" // Use the header's profile-specific GLSL exporter.
#include <iostream> // Read shader text and print exported text or an error.
#include <iterator> // Read standard input through character iterators.
int main(int argc, char** argv) { // Export one profile named on the command line.
    if (argc != 2) { // Require exactly one profile argument.
        return 1; // Report incorrect command-line usage.
    } // The remaining code has a valid argument count.
    const std::string name(argv[1]); // Keep the requested profile name.
    shdr::profile selected = shdr::profile::gl46; // Preserve the default for unrecognized names.
    if (name == "es31") { // Select the minimum OpenGL ES compute profile.
        selected = shdr::profile::es31; // Export GLSL ES 3.10.
    } else if (name == "es32") { // Select the newer OpenGL ES profile.
        selected = shdr::profile::es32; // Export GLSL ES 3.20.
    } else if (name == "gl43") { // Select the minimum desktop compute profile.
        selected = shdr::profile::gl43; // Export desktop GLSL 4.30.
    } else if (name == "gl45") { // Select desktop OpenGL 4.5.
        selected = shdr::profile::gl45; // Export desktop GLSL 4.50.
    } // Other names keep the original OpenGL 4.6 fallback.
    const std::string source( // Read the complete shader from standard input.
        (std::istreambuf_iterator<char>(std::cin)), // Start at the first input character.
        std::istreambuf_iterator<char>()); // Stop when standard input reaches its end.
    try { // Convert export failures into a separate helper error status.
        std::cout << shdr::glsl_source(source, selected); // Print standalone GLSL for this profile.
    } catch (const std::exception& error) { // Capture a rejected source or export configuration.
        std::cerr << error.what(); // Preserve the reason for the Python compiler report.
        return 2; // Distinguish export errors from incorrect argument counts.
    } // A successful export reaches the normal success return.
} // Return zero implicitly after successful export.
"""
            )  # Write the complete temporary C++11 exporter source.
            options.exporter = Path(export_directory) / "export"  # Use the temporary exporter built for this header snapshot.
            checked(  # Build the temporary exporter before profile checks.
                [  # List the strict compiler arguments for the exporter.
                    options.cxx,  # Use the requested C++ compiler.
                    "-std=c++11",  # Require the library's minimum C++ language standard.
                    "-pedantic-errors",  # Treat nonstandard C++ syntax as an error.
                    "-pthread",  # Link the runtime's thread support into the temporary exporter.
                    "-I",  # Add the runtime header directory to the include search path.
                    str(ROOT),  # Read the runtime header from this checkout.
                    str(exporter_source),  # Compile the temporary runtime-export helper.
                    "-o",  # Choose the temporary executable output path.
                    str(options.exporter),  # Store the helper in this temporary directory.
                ]  # Complete the strict C++11 exporter build command.
            )  # Require a working exporter before starting profile checks.
            with ThreadPoolExecutor(max_workers=options.jobs) as pool:  # Limit concurrent checks to the requested worker count.
                records = list(  # Collect one completed result per selected source.
                    pool.map(lambda case: verify_case(case, cache, options, frontend), cases)  # Check each source independently while preserving selected-case order.
                )  # Materialize ordered results before the worker pool closes.
        if (  # Publish only results obtained from one unchanged frontend and header snapshot.
            frontend_bytes != (ROOT / "scripts" / "compile_shader.py").read_bytes()  # Reject results if the frontend changed while workers ran.
            or header_bytes != (ROOT / "shdr.h").read_bytes()  # Reject results if the runtime header changed during the checks.
        ):  # Detect any source snapshot change during the parallel checks.
            raise RuntimeError(  # Require every result to use the same source snapshot.
                "frontend or header changed during verification. Rerun against a stable build"  # Explain why a stable frontend and header are required.
            )  # Stop before writing an inconsistent report.
        report = {  # Assemble provenance, counts and detailed compiler outcomes.
            "schema_version": 1,  # Identify the report layout version.
            "repositories": REPOSITORIES,  # Record the pinned source collections.
            "tools": {  # Record the compiler versions used for these results.
                "glslang": checked([options.glslang, "--version"]).splitlines()[0],  # Record the reference GLSL compiler version.
                "cxx": checked([options.cxx, "--version"]).splitlines()[0],  # Record the strict C++11 compiler version.
            },  # Complete the compiler-version record.
            "frontend_sha256": digest(frontend_bytes),  # Identify the frontend used for CPU adaptation.
            "header_sha256": digest(header_bytes),  # Identify the runtime used for GPU source export.
            "method": {  # Describe how the compiler checks were performed.
                "selection": "all tracked .comp files in the pinned paths, excluding filenames identifying HLSL",  # Describe which tracked files were selected.
                "reference": "original standalone GLSL files or explicitly configured unchanged NCNN bodies. OpenGL compile/link, then Vulkan SPIR-V fallback",  # Describe the unmodified-source reference check.
                "cpu": "current generated adapter compiled as standard C++11. No shader execution",  # Describe the generated C++11 check.
                "generated_gpu": "embedded source exported by the header and compiled/linked for each supported profile",  # Describe the exported GLSL profile checks.
                "excluded_hlsl": excluded_hlsl,  # Count HLSL inputs separately from the GLSL collection.
            },  # Complete the description of the reference and generated checks.
            "summary": summary(records),  # Store collection and repository totals.
            "results": records,  # Store all individual source records.
        }  # Complete the provenance and all individual source outcomes.
        options.output.parent.mkdir(parents=True, exist_ok=True)  # Create the report directory if needed.
        options.output.write_text(render_public_report(report), encoding="utf-8")  # Preserve every source record and fingerprint in a readable HTML report.
        print(json.dumps(report["summary"], indent=2))  # Print the aggregate counters for a quick review.
        print("Results: " + str(options.output))  # Print the path to the detailed machine-readable results.
    except (OSError, RuntimeError) as error:  # Report this failure through the command's normal error path.
        parser.exit(1, str(error) + "\n")  # Report the failure and exit with a nonzero status.


if __name__ == "__main__":  # Run only when this file is invoked as a command.
    main()  # Read options and run the requested checks.
