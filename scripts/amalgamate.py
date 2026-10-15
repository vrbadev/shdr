#!/usr/bin/env python3
"""Rebuild the dependency-free public header from its maintained sections."""
from pathlib import Path  # Read and write the source files.
import argparse  # Read the optional check flag.

root = Path(__file__).resolve().parents[1]  # Find the checkout containing detail/.
parser = argparse.ArgumentParser(description=__doc__)  # Describe this tool in --help.
parser.add_argument("--check", action="store_true")  # Support a read-only comparison mode.
args = parser.parse_args()  # Read the requested mode.
parts = [  # Start the output with its source note and include guard.
    "// shdr 1.0.0\n",  # Identify the release directly in the distributed header.
    "// C++11 CPU and OpenGL compute pipelines.\n",  # Explain where to edit.
    "#ifndef SHDR_SINGLE_HEADER_INCLUDED\n#define SHDR_SINGLE_HEADER_INCLUDED\n",  # Avoid duplicate definitions.
]  # The maintained sections are appended next.
for name in (  # Append each maintained source section in dependency order.
    "math.hpp",  # Define portable numeric types before the runtime uses them.
    "runtime.hpp",  # Define resources, kernels, devices and pipeline execution.
    "async.hpp",  # Extend those pipelines with completion handles and CPU workers.
    "egl.hpp",  # Add the optional headless device after its base runtime is defined.
    "macros.hpp",  # Declare adapter macros after all underlying runtime types.
):  # Keep dependency order.
    parts.append("\n// ---- " + name + " ----\n")  # Mark the start of each section.
    parts.append((root / "detail" / name).read_text())  # Copy the complete section.
parts.append("\n#endif // SHDR_SINGLE_HEADER_INCLUDED\n")  # Close the public include guard.
output = "".join(parts)  # Assemble the header from its exact source sections.
if args.check:  # Compare the assembled header with the committed file.
    if (root / "shdr.h").read_text() != output:  # Detect any missing regeneration.
        raise SystemExit("shdr.h is stale. Run scripts/amalgamate.py")  # Explain the repair.
else:  # Normal mode writes the assembled header.
    (root / "shdr.h").write_text(output)  # Replace the generated file with current sections.
