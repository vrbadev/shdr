#!/usr/bin/env python3
"""Render complete source files as versioned, linked HTML pages.

Each page includes readable code, numbered lines and a file download.
"""

from __future__ import annotations  # Allow readable type hints with Python 3.8.

import base64  # Embed each file download in its source page.
import hashlib  # Identify the exact source shown on each page.
import html  # Escape code and filenames before adding them to HTML.
import os  # Calculate links relative to each nested source page.
import re  # Read the release label from the documented runtime header.
from pathlib import Path  # Discover published sources and write their HTML pages.
from highlight_docs import add_stylesheet, highlight_lines, language_for_path  # Render static syntax colors with the source file's actual language.

ROOT = Path(__file__).resolve().parent.parent  # Locate the checkout containing this script.
DOCS = ROOT / "docs"  # Keep all published pages under the documentation root.
VERSION_MATCH = re.search(r'^\s*#define\s+SHDR_VERSION\s+"([^"]+)"', (ROOT / "shdr.h").read_text(encoding="utf-8"), re.MULTILINE)  # Use the same version as the published runtime.
if not VERSION_MATCH:  # Refuse to publish a source browser for an unidentified release.
    raise SystemExit("SHDR_VERSION is missing from shdr.h")  # Explain which release definition is needed.
VERSION = html.escape(VERSION_MATCH.group(1))  # Escape the release label before adding it to page chrome.


def navigation(prefix: str) -> str:  # Build the same navigation used by the guides.
    """Return a keyboard-accessible page header with local documentation links."""
    links = (("index.html", "Start"), ("examples.html", "Examples"), ("results.html", "Results"), ("corpus.html", "Public shaders"), ("api.html", "API"))  # Keep the main destinations in reading order.
    items = "".join('<a href="' + prefix + target + '">' + label + '</a>' for target, label in links)  # Create links relative to the current page.
    return '<a class="skip-link" href="#content">Skip to content</a><header class="site-header"><div class="header-inner"><a class="brand" href="' + prefix + 'index.html">shdr<span class="release-version">' + VERSION + '</span><span>CPU + GPU compute</span></a><nav class="site-nav" aria-label="Main navigation">' + items + '</nav></div></header>'  # Include the documented release and a visible keyboard skip link.


def page(title: str, body: str, target: Path) -> str:  # Wrap one source document in the shared page shell.
    """Create a complete source page with local documentation assets."""
    prefix = Path(os.path.relpath(DOCS, target.parent)).as_posix().rstrip("/") + "/"  # Find the documentation root from this page.
    return '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Read and download the documented shdr example and library sources."><title>' + html.escape(title) + ' · shdr ' + VERSION + '</title><link rel="stylesheet" href="' + prefix + 'site.css"></head><body>' + navigation(prefix) + '<div class="page-shell"><main class="page-main source-view" id="content">' + body + '</main></div><footer class="footer"><p>shdr ' + VERSION + ' · Source browser · <a href="' + prefix + 'examples.html">Choose an example</a> · <a href="' + prefix + 'portability.html">Portability and requirements</a></p></footer></body></html>\n'  # Keep the entire shell local to this site.


def published_sources() -> list[Path]:  # Select readable files that are useful when learning the library.
    """List example programs, shader bodies, wrappers and maintained build tools."""
    files = [ROOT / "shdr.h", ROOT / "CMakeLists.txt", ROOT / "cmake" / "shdr.cmake"]  # Include the single runtime header and build integration.
    files.extend(sorted((ROOT / "detail").glob("*.hpp")))  # Explain where the amalgamated runtime is maintained.
    files.extend(sorted((ROOT / "examples").glob("*.cpp")))  # Include every complete host application and plugin.
    files.extend(sorted((ROOT / "examples").glob("*.hpp")))  # Include reusable pipeline wrappers and independent references.
    files.extend(sorted((ROOT / "examples" / "shaders").glob("*.cpp")))  # Include all native shader sources.
    files.extend(sorted((ROOT / "scripts").glob("*.py")))  # Include import, validation and dataset preparation tools.
    files.extend(sorted((ROOT / "tests").glob("*.cpp")))  # Include independent runtime, math and GPU regression tests.
    files.extend(sorted((ROOT / "tests").glob("*.py")))  # Include native shader import and compilation regressions.
    data_readme = ROOT / "examples" / "data" / "readme.md"  # Find the human-readable data provenance maintained with the fixtures.
    if data_readme.exists():  # Include the provenance when it is available.
        files.append(data_readme)  # Make its content readable within the published site.
    return sorted(set(path for path in files if path.is_file()))  # Return a stable list of unique paths.


def write_source(path: Path) -> Path:  # Publish one exact source file as readable HTML.
    """Write numbered source lines and a byte-exact embedded download."""
    relative = path.relative_to(ROOT)  # Show paths relative to the checkout.
    target = DOCS / "source" / (relative.as_posix() + ".html")  # Preserve the familiar source directory structure.
    payload = path.read_bytes()  # Keep the original bytes for downloads and checksums.
    text = payload.decode("utf-8")  # All selected source files use UTF-8 text.
    digest = hashlib.sha256(payload).hexdigest()  # Identify the exact version displayed here.
    download = base64.b64encode(payload).decode("ascii")  # Encode the original file as a browser download.
    lines = []  # Collect the numbered source lines.
    rendered = highlight_lines(text, language_for_path(path))  # Tokenize the full file so multiline comments and strings retain their state.
    for number, line in enumerate(rendered[: len(text.splitlines())], 1):  # Keep the same numbered lines as the original source browser.
        lines.append(f'<span class="line" id="L{number}"><a class="line-number" href="#L{number}" aria-label="Line {number}">{number}</a>{line}</span>')  # Keep balanced escaped token spans inside each source line.
    body = '<p class="eyebrow">Source browser</p><h1><code>' + html.escape(relative.as_posix()) + '</code></h1><p>Read the complete file below or download its original bytes. Line numbers provide stable links to a declaration or processing step.</p><div class="actions"><a class="button-link" href="' + Path(os.path.relpath(DOCS / "source" / "index.html", target.parent)).as_posix() + '">All source files</a><a class="button-link" id="download" download="' + html.escape(path.name, quote=True) + '" href="data:application/octet-stream;base64,' + download + '">Download ' + html.escape(path.name) + '</a></div><details><summary>File fingerprint</summary><p class="metadata">SHA-256 <code>' + digest + '</code></p></details><pre class="source-code"><code>' + "\n".join(lines) + '</code></pre>'  # Keep the source and its exact download together.
    target.parent.mkdir(parents=True, exist_ok=True)  # Create this source directory when needed.
    css = Path(os.path.relpath(DOCS / "highlight.css", target.parent)).as_posix()  # Find the local syntax stylesheet from this nested source page.
    target.write_text(add_stylesheet(page(relative.as_posix(), body, target), css), encoding="utf-8")  # Write a complete standalone page with static syntax colors.
    return target  # Return the generated page for the index.


def main() -> None:  # Generate every source page and its searchable index.
    """Render the source collection from the maintained source files."""
    files = published_sources()  # Gather all current published text files.
    generated = [write_source(path) for path in files]  # Render each file with its exact contents.
    expected = set(generated) | {DOCS / "source" / "index.html"}  # Keep the current source pages and their generated index.
    for previous in (DOCS / "source").rglob("*.html"):  # Inspect only the source browser's generated pages.
        if previous not in expected:  # Find a page for a source removed from the published collection.
            previous.unlink()  # Remove the obsolete generated source page.
    items = []  # Collect index links grouped by their normal source path.
    for source, target in zip(files, generated):  # Pair source paths with their generated browser pages.
        relative = source.relative_to(ROOT).as_posix()  # Use a machine-independent path as the label.
        items.append('<li data-file="' + html.escape(relative.lower(), quote=True) + '"><a href="' + target.relative_to(DOCS / "source").as_posix() + '"><code>' + html.escape(relative) + '</code></a></li>')  # Give search a plain lowercase filename.
    body = '<p class="eyebrow">Complete files</p><h1>Read the source</h1><p class="lead">Follow a complete application, inspect one shader, or look up a reusable pipeline wrapper. Every file has line links and an exact download.</p><p>Start with <a href="../examples.html">the example catalog</a> for a guided choice. Native shader files are converted during the build. Their C++ adapters and portable GPU variants are embedded in the application.</p><label for="file-search">Find a source file</label><input type="search" id="file-search" placeholder="Try affine, fft, lenet or data_sources"><p id="file-count" class="muted" aria-live="polite">' + str(len(files)) + ' files</p><ul class="file-list">' + "\n".join(items) + '</ul><script>\n// Filter the existing file links without requesting any network resources.\nconst query = document.getElementById("file-search");\nconst files = Array.from(document.querySelectorAll("[data-file]"));\nquery.addEventListener("input", () => {\n  const words = query.value.toLowerCase().trim().split(/\\s+/).filter(Boolean); // Match every search word.\n  let count = 0; // Count the visible files.\n  files.forEach(file => {\n    file.hidden = !words.every(word => file.dataset.file.includes(word)); // Preserve the original file order.\n    if (!file.hidden) count++; // Report the number of visible links.\n  });\n  document.getElementById("file-count").textContent = count + " files"; // Announce the updated count.\n});\n</script>'  # Keep all files visible when JavaScript is unavailable.
    index = DOCS / "source" / "index.html"  # Keep the browser entry point beside its generated files.
    index.write_text(page("Read the source", body, index), encoding="utf-8")  # Publish the searchable file list.
    print("Published " + str(len(files)) + " source files")  # Report the completed source collection.


if __name__ == "__main__":  # Run only when invoked as a command.
    main()  # Rebuild the source browser from the current checkout.
