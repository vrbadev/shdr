#!/usr/bin/env python3
"""Export every dual shader and validate all supported GLSL language profiles."""

import argparse  # Read the exporter, compiler and source paths.
from pathlib import Path  # Build paths with pathlib.
import subprocess  # Run the exporter and GLSL compiler.
import tempfile  # Remove exported shader files after validation.


def main():  # Validate every bundled compute shader for every supported profile.
    parser = argparse.ArgumentParser(description=__doc__)  # Provide command help.
    parser.add_argument("--exporter", type=Path, required=True)  # Locate the built shader exporter.
    parser.add_argument("--validator", type=Path, required=True)  # Locate glslangValidator.
    parser.add_argument("--source-dir", type=Path, required=True)  # Locate the shader examples.
    options = parser.parse_args()  # Read all required paths.
    shaders = sorted(  # Collect bundled shader filenames in reproducible order.
        (options.source_dir / "examples" / "shaders").glob("*.cpp")  # Read native compute sources in the example directory.
    )  # Use a stable order.
    if not shaders:  # Validate the selected source directory.
        parser.error("no shaders found")  # Stop with a useful path-related error.
    profiles = ("es31", "es32", "gl43", "gl45", "gl46")  # Cover all runtime language choices.
    with tempfile.TemporaryDirectory(  # Remove all exported source files after the checks.
        prefix="shdr-glsl-"  # Identify temporary shader validation directories.
    ) as directory:  # Keep exports outside the source tree.
        for shader in shaders:  # Visit each registered shader factory.
            for profile in profiles:  # Export it using each version preamble.
                command = [  # Build this factory's profile-specific export command.
                    str(options.exporter),  # Use the built executable containing embedded shader text.
                    shader.stem + "_kernel",  # Name the factory registered for this source file.
                    profile,  # Select the exported shader language version.
                ]  # Select factory and profile.
                result = subprocess.run(  # Capture the exact text embedded in the executable.
                    command,  # Pass each compiler argument directly.
                    check=True,  # Propagate shader export failures.
                    stdout=subprocess.PIPE,  # Read the exported source from standard output.
                    text=True,  # Decode compiler output as text.
                )  # The exporter result now contains the source.
                path = Path(directory) / (  # Choose one temporary output path per source and profile.
                    shader.stem + "_" + profile + ".comp"  # Keep compiler diagnostics easy to associate with their input.
                )  # Give each check a separate file.
                path.write_text(  # Preserve the exact standalone source emitted by the runtime.
                    result.stdout, encoding="utf-8"  # Write shader text with a deterministic encoding.
                )  # Write standard GLSL for the compiler.
                command = [  # Build the standard GLSL compiler command.
                    str(options.validator),  # Use the configured standalone validator.
                    "-S",  # Select a shader stage independently of the filename suffix.
                    "comp",  # Require compute-shader grammar and built-ins.
                    str(path),  # Compile this exported source/profile pair.
                ]  # Select the compute shader stage.
                subprocess.run(command, check=True)  # Stop on the first language error.
    total = len(shaders) * len(profiles)  # Count all successful source/profile pairs.
    message = "Validated {} shaders for {} profiles ({} compilations)"  # Keep the summary readable.
    print(message.format(len(shaders), len(profiles), total))  # Report the completed checks.


if __name__ == "__main__":  # Run only when this file is used as a command.
    main()  # Read options and start validation.
