#!/usr/bin/env python3
"""Render public runtime declarations as a searchable HTML reference.

The reference includes syntax-highlighted signatures and linked header source.
"""

from __future__ import annotations  # Keep type hints usable on older supported Python versions.

import argparse  # Read command-line options and show help.
import html  # Escape source text before placing it in HTML.
from pathlib import Path  # Build paths and read or write files.
import re  # Find declarations, comments and diagnostic lines.
from highlight_docs import add_stylesheet, highlight_code, highlight_lines  # Render static syntax colors and preserve source text.


ROOT = Path(__file__).resolve().parent.parent  # Find the checkout containing the maintained source files.
GROUPS = (  # List the API sections in their displayed order.
    "Build integration and shader exports",  # Group device setup and shader export helpers.
    "Resources and synchronization",  # Group shared buffers, images and transfer methods.
    "Kernels and dispatch",  # Group kernel interfaces and dispatch dimensions.
    "Pipelines and dynamic libraries",  # Group linked stages and kernel factories.
    "Asynchronous execution",  # Group fence handles and CPU work queues.
    "Headless EGL backend",  # Group optional context and headless device creation.
    "GLSL math",  # Group portable vector, matrix and numeric functions.
    "C++ adapter declaration macros",  # Group the shader interface declaration macros.
    "Other documented declarations",  # Keep remaining public helpers discoverable.
)  # Keep this display order when collecting rendered API sections.


def clean_comment(raw: str) -> str:  # Remove C++ comment prefixes before rendering their text.
    """Strip comment decoration while preserving paragraph boundaries."""
    raw = re.sub(r"^/\*\*|\*/$", "", raw.strip())  # Remove the outer comment markers.
    return "\n".join(  # Return comment text with its paragraph boundaries preserved.
        re.sub(r"^\s*(?:\*\s?|///\s?)", "", line).rstrip() for line in raw.splitlines()  # Remove decoration from each comment line.
    ).strip()  # Remove whitespace around the complete cleaned docstring.


def declaration_after(source: str, end: int) -> str:  # Read the C++ declaration immediately following a docstring.
    """Read a declaration, omitting inline bodies and multiline macro bodies."""
    remainder = source[end:].lstrip()  # Read the text following this declaration or field.
    if not remainder or remainder.startswith(("/**", "///")):  # Select comments attached to a declaration.
        return ""  # Return an empty signature for this unattached comment.
    if remainder.startswith("#define"):  # Show the macro name and parameters as its signature.
        macro = re.match(r"#define\s+\w+(?:\([^)]*\))?", remainder)  # Recognize a generated macro or declaration name.
        return macro.group() if macro else remainder.splitlines()[0]  # Keep the macro name and parameters, or its first line as a fallback.
    if re.match(r"enum(?:\s+class)?\s+\w+", remainder):  # Keep the complete value list when the documented declaration is an enum.
        end = remainder.find(";")  # Locate the declaration or directive end.
        if 0 <= end < 800:  # Accept a nearby enum terminator while avoiding an unrelated later declaration.
            return remainder[: end + 1]  # Include the enum declaration through its final semicolon.
    lines = []  # Collect declaration signature lines.
    for line in remainder.splitlines()[:14]:  # Limit the search to fourteen lines after the docstring.
        line = line.split("//", 1)[0].rstrip()  # Remove inline C++ comments from the displayed signature.
        found = re.search(r"[;{]", line)  # Locate the first declaration terminator on this line.
        if found:  # Stop at a declaration terminator or the opening of an inline function body.
            lines.append(line[: found.start()] + (";" if line[found.start()] == ";" else ""))  # Keep this line in the displayed C++ signature.
            break  # Stop once the declaration boundary has been found.
        if line.strip():  # Retain nonempty signature lines before that boundary.
            lines.append(line)  # Keep this line in the displayed C++ signature.
    return "\n".join(lines).strip()  # Return the collected signature with its original line breaks.


def choose_group(declaration: str, comment: str, position: int, source: str) -> str:  # Place each API entry in a useful documentation section.
    """Group public types and functions, with a GLSL namespace fallback."""
    combined = declaration + " " + comment  # Search both the signature and its explanation.
    if re.match(r"#define\s+SHDR_VERSION(?:_|\b)", declaration):  # Keep release identification with the basic build and setup helpers.
        return GROUPS[0]  # Show version macros before the shader declaration interface.
    if declaration.startswith("#define"):  # Keep declaration macros together even when their prose mentions resources.
        return GROUPS[7]  # Use the adapter macro section.
    if re.search(r"\b(headless_context|headless_device|backend_unavailable)\b", combined):  # Recognize the optional EGL public types by name.
        return GROUPS[5]  # Use the EGL section.
    async_start = source.find("// ---- async.hpp ----")  # Locate the asynchronous runtime section.
    egl_start = source.find("// ---- egl.hpp ----")  # Locate the optional EGL section.
    macros_start = source.find("// ---- macros.hpp ----")  # Locate the adapter declaration section.
    if egl_start >= 0 and position >= egl_start and (macros_start < 0 or position < macros_start):  # Place declarations within the EGL source section in its own group.
        return GROUPS[5]  # Use the EGL section.
    if async_start >= 0 and position >= async_start and (egl_start < 0 or position < egl_start):  # Place declarations within the asynchronous source section in its own group.
        return GROUPS[4]  # Use the asynchronous section.
    if re.search(  # Recognize pipeline and registered factory declarations.
        r"\b(pipeline|module|plugin|kernel_export|shared_library|load_kernel|make_kernel|register_kernel)\b",  # Recognize pipeline composition and dynamic factory names.
        combined,  # Search the signature together with its explanation.
    ):  # Use composition and factory names to choose the pipeline group.
        return GROUPS[3]  # Use the pipeline section.
    if re.search(r"\b(stage|dispatch|backend|kernel)\b", declaration):  # Recognize dispatch dimensions, stages and kernel interfaces.
        return GROUPS[2]  # Use the kernel section.
    if re.search(  # Recognize resource types and transfer methods.
        r"\b(buffer|image2d|access|upload|download|from_gl|host_dirty|gpu_dirty)\b", combined  # Match the supported names in this API group.
    ):  # Group buffer and image synchronization APIs together.
        return GROUPS[1]  # Use the resource section.
    if re.search(  # Recognize basic setup, source export and error declarations.
        r"\b(device|profile|glsl_source|glsl_prelude|read_file|error|resolver|extent|ceil_div)\b",  # Recognize device setup, shader export and basic runtime names.
        combined,  # Search the signature together with its explanation.
    ):  # Group setup helpers with backend configuration.
        return GROUPS[0]  # Use the build and device section.
    glsl_start = source.find("namespace glsl")  # Locate the math namespace in the header.
    runtime_start = source.find("namespace shdr", glsl_start + 1)  # Find where runtime declarations follow the math section.
    if (  # Use namespace position to identify remaining math declarations.
        glsl_start >= 0  # Require that the math namespace was found.
        and position >= glsl_start  # Check that this declaration follows the math namespace opening.
        and (runtime_start < 0 or position < runtime_start)  # Keep math entries before the runtime namespace begins.
    ):  # Keep declarations inside the math namespace in the math group.
        return GROUPS[6]  # Use the math section.
    return GROUPS[8]  # Use the remaining public declarations section.


def family_symbols(body: str) -> list[str]:  # List functions covered by one math-family docstring.
    """Collect names from public declarations and the header's math generator calls."""
    body = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", body)  # Remove comments before scanning math declarations.
    names = []  # Collect the public names found in this family.
    depth = 0  # Track nested bodies so only public names are listed.
    in_macro = False  # Track whether this line continues a macro definition.
    for line in body.splitlines():  # Read each declaration and generator call in source order.
        stripped = line.strip()  # Inspect the current line after trimming indentation.
        if in_macro or stripped.startswith("#define"):  # Select public declarations surrounding the macro definitions.
            in_macro = stripped.endswith("\\")  # Track whether this line continues a macro definition.
            continue  # Skip this item and inspect the next one.
        macro = re.match(r"SHDR_DETAIL_\w+\(([^,)]+)", stripped)  # Recognize a generated macro or declaration name.
        if macro and depth == 0:  # Read a public symbol passed to a top-level math generator.
            name = macro.group(1).strip()  # Trim surrounding whitespace from the symbol name.
            if re.fullmatch(r"[A-Za-z_]\w*", name):  # Accept ordinary identifier names from a generator call.
                names.append(name)  # Keep this public function or operator name.
            elif re.fullmatch(r"[+*/%&|^<>=!-]+", name):  # Handle this source form.
                names.append("operator" + name)  # Recognize operator tokens supplied to the math generator.
        elif depth == 0 and not stripped.startswith("#"):  # Handle this source form.
            candidates = re.findall(r"(?<![:\w])([A-Za-z_]\w*)\s*\(", line)  # Scan ordinary top-level declarations outside preprocessor directives.
            candidates = [  # Find possible function names and remove language keywords.
                name  # Keep this candidate function name.
                for name in candidates  # Inspect the names matched on this declaration line.
                if name  # Keep candidate public function names.
                not in {"if", "for", "while", "switch", "static_assert", "decltype", "sizeof"}  # Remove language constructs that resemble function calls.
            ]  # Remove language keywords from the list of possible function names.
            if candidates:  # Keep the first public function name on this declaration line.
                names.append(candidates[0])  # Keep this public function or operator name.
        depth += line.count("{") - line.count("}")  # Track nested bodies so only public names are listed.
    return list(dict.fromkeys(names))  # Return unique public names in source order.


def documented_declarations(source: str) -> list[dict[str, str | int]]:  # Collect public declarations and their source locations.
    """Collect header docstrings, declarations, groups, and source line numbers."""
    expression = re.compile(r"/\*\*[\s\S]*?\*/|(?:^[ \t]*///[^\n]*(?:\n|$))+", re.MULTILINE)  # Match block docstrings and triple-slash comments.
    entries = []  # Collect the documented public declarations.
    for match in expression.finditer(source):  # Visit each matched public docstring.
        comment = clean_comment(match.group())  # Read the cleaned C++ comment text.
        if not comment or "@internal" in comment or "\\internal" in comment:  # Skip empty comments and entries explicitly marked internal.
            continue  # Continue to the next public docstring.
        if re.fullmatch(r"(?:@|\\)[{}]\s*", comment):  # Recognize structural Doxygen group markers.
            continue  # Continue to the next documented declaration.
        family = re.search(r"(?:@|\\)name\s+([^\n]+)", comment)  # Recognize a named group of math functions.
        if family:  # Treat a named Doxygen family as one math API entry.
            closing = re.search(r"/\*\*\s*(?:@|\\)}\s*\*/", source[match.end() :])  # Locate the end of the documented math family.
            family_end = match.end() + closing.start() if closing else match.end()  # Limit the symbol scan to this family.
            names = family_symbols(source[match.end() : family_end])  # Collect the public names found in this family.
            declaration = family.group(1) + ("\n" + ", ".join(names) if names else "")  # Read the displayed signature for this API entry.
            entries.append(  # Store the public signature, explanation, group and source line.
                {  # Store the location needed to link this entry to its header.
                    "comment": comment,  # Keep the entry's explanatory text.
                    "declaration": declaration,  # Keep the displayed C++ signature.
                    "group": GROUPS[6],  # Choose the API section for this entry.
                    "line": source.count("\n", 0, match.start()) + 1,  # Link this entry to its header source line.
                }  # Complete this family entry with its source location.
            )  # Add the entire documented family as one API card.
            continue  # Continue after publishing this complete family entry.
        declaration = declaration_after(source, match.end())  # Read the displayed signature for this API entry.
        if not declaration or re.match(r"(?:#(?:endif|if|include)|namespace\b)", declaration):  # Skip unattached comments and namespace or conditional directives.
            continue  # Continue to the next attached public declaration.
        entries.append(  # Store the public signature, explanation, group and source line.
            {  # Store the location needed to link this entry to its header.
                "comment": comment,  # Keep the entry's explanatory text.
                "declaration": declaration,  # Keep the displayed C++ signature.
                "group": choose_group(declaration, comment, match.start(), source),  # Choose the API section for this entry.
                "line": source.count("\n", 0, match.start()) + 1,  # Link this entry to its header source line.
            }  # Complete the ordinary declaration entry.
        )  # Add this declaration to the exported API entries.
    return entries  # Return all collected public API entries.


def render_comment(comment: str) -> str:  # Convert the supported Doxygen tags to escaped HTML.
    """Render a small Doxygen tag subset, escaping every source-content string."""
    comment = re.sub(r"(?:@|\\)brief\s+", "", comment)  # Remove the brief tag while retaining its explanation.
    comment = re.sub(r"(?:@|\\)(?:ingroup|defgroup)\s+[^\n]+", "", comment)  # Drop group-control tags from the displayed prose.
    comment = re.sub(r"(?:@|\\)name\s+[^\n]+", "", comment)  # Keep the family title in its signature.
    comment = re.sub(r"(?:@|\\)[{}]", "", comment)  # Remove Doxygen group-boundary markers from the prose.
    paragraphs = []  # Collect escaped HTML paragraphs.
    for paragraph in re.split(r"\n\s*\n", comment.strip()):  # Render each comment paragraph separately.
        escaped = html.escape(paragraph)  # Escape text before adding supported formatting tags.
        escaped = re.sub(  # Turn a parameter tag into a label while retaining its explanation.
            r"(?:@|\\)param(?:\[[^]]+\])?\s+(\w+)\s+",  # Match an optional direction qualifier and the parameter name.
            r"<strong>Parameter <code>\1</code>:</strong> ",  # Show each parameter name beside its explanation.
            escaped,  # Keep the escaped explanation in the displayed paragraph.
        )  # Finish rendering this parameter label.
        for tags, label in (  # Handle each supported Doxygen label.
            ("return|returns", "Returns"),  # Label documented return values.
            ("throw|throws|exception", "Throws"),  # Label documented exceptions.
            ("note", "Note"),  # Label additional usage notes.
            ("warning", "Requirement"),  # Label required preconditions.
        ):  # Apply each label to the matching Doxygen tag.
            escaped = re.sub(  # Replace the current tag with a readable HTML label.
                r"(?:@|\\)(?:" + tags + r")\s+", "<strong>" + label + ":</strong> ", escaped  # Replace this documented tag and preserve surrounding text.
            )  # Keep the rest of this escaped paragraph unchanged.
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)  # Render backtick-delimited text as inline code.
        paragraphs.append("<p>" + escaped.replace("\n", "<br>") + "</p>")  # Keep this escaped paragraph in its original order.
    return "\n".join(paragraphs)  # Return the joined HTML paragraphs.


def export(header: Path, output: Path) -> int:  # Build the searchable API page from the public header.
    """Write deterministic HTML and return the number of exported declarations."""
    source = header.read_text(encoding="utf-8")  # Read the complete amalgamated header as UTF-8 text.
    version_match = re.search(r'^\s*#define\s+SHDR_VERSION\s+"([^"]+)"', source, re.MULTILINE)  # Read the release label from the actual exported header.
    if not version_match:  # Require the header to identify the documented release.
        raise SystemExit("SHDR_VERSION is missing from " + str(header))  # Avoid publishing an unlabeled API reference.
    version = version_match.group(1)  # Keep this release label separate from all API source text.
    entries = documented_declarations(source)  # Collect the documented public declarations.
    if not entries:  # Require public API comments in the selected header.
        raise SystemExit("No API docstrings found in " + str(header))  # Identify the header that needs public API comments.
    navigation = []  # Collect links to the visible API sections.
    sections = []  # Collect the rendered API groups.
    for index, group in enumerate(GROUPS):  # Render each API group in its displayed order.
        selected = [entry for entry in entries if entry["group"] == group]  # Visit groups in the order shown by the navigation.
        if not selected:  # Select groups containing public entries.
            continue  # Skip this item and inspect the next one.
        anchor = "group-" + str(index)  # Continue to the next populated group.
        navigation.append(f'<a href="#{anchor}">{html.escape(group)}</a>')  # Link the displayed section from the page navigation.
        articles = []  # Collect each declaration card in this section.
        for entry in selected:  # Render each declaration in this group.
            signature = str(entry["declaration"])  # Read this entry's displayed C++ declaration.
            query = re.sub(r"\s+", " ", signature + " " + str(entry["comment"])).lower()  # Normalize the entry text for case-insensitive search.
            articles.append(  # Add one escaped declaration card to this API section.
                '<article class="entry" data-search="' + html.escape(query, quote=True) + '">\n'  # Keep source text escaped in the HTML card.
                '<pre><code class="syntax-highlight">'  # Keep each highlighted declaration in its own readable code block.
                + highlight_code(signature, "cpp")  # Color the C++ declaration while preserving its exact displayed text.
                + "</code></pre>\n"  # Close the displayed signature before its explanation.
                + render_comment(str(entry["comment"]))  # Show the formatted docstring below its declaration.
                + "\n"  # Separate the rendered pieces with readable line breaks.
                + f'<a class="source" href="#L{entry["line"]}">shdr.h, line {entry["line"]}</a>\n</article>'  # Link the API card to its declaration in the source browser.
            )  # Finish this part of the API page.
        sections.append(  # Add the rendered group to the complete page.
            f'<section id="{anchor}"><h2>{html.escape(group)}</h2>\n'  # Start the section with a stable anchor and escaped group title.
            + "\n".join(articles)  # Place this group's declaration cards below its title.
            + "\n</section>"  # Close the section after its final declaration card.
        )  # Finish this part of the API page.
    document = (  # Assemble the complete standalone HTML page.
        """<!doctype html>
<html lang="en"> <!-- Declare the language used by the documentation. -->
<head> <!-- Keep page metadata and styles separate from its content. -->
<meta charset="utf-8"> <!-- Display source names and math symbols as UTF-8. -->
<meta name="viewport" content="width=device-width,initial-scale=1"> <!-- Fit small screens. -->
<title>shdr __SHDR_DOC_VERSION__ API reference</title> <!-- Name the browser tab and release. -->
<style> /* Keep the API page usable without a stylesheet download. */
:root { /* Set colors for the default light theme. */
    color-scheme: light dark; /* Allow matching browser controls in either theme. */
    --bg: #fbfcff; /* Use a light page background. */
    --fg: #1f2937; /* Use dark text with readable contrast. */
    --border: #d7dfe9; /* Separate API entries with a subtle rule. */
    --accent: #1d4ed8; /* Give links a recognizable blue color. */
    --code: #eef2f8; /* Set code apart from prose. */
} /* Finish the light-theme values. */
@media (prefers-color-scheme: dark) { /* Follow the reader's system theme. */
    :root { /* Override only the theme colors. */
        --bg: #111827; /* Use a dark page background. */
        --fg: #e5e7eb; /* Keep text readable on the dark background. */
        --border: #374151; /* Keep entry rules visible without high contrast. */
        --accent: #93c5fd; /* Lighten links for the dark background. */
        --code: #1f2937; /* Keep code panels distinct from the page. */
    } /* Finish the dark color overrides. */
} /* Keep layout rules shared by both themes. */
* { box-sizing: border-box; } /* Include padding in element widths. */
body { /* Use a readable system font and remove default page margins. */
    margin: 0; /* Let the main content control spacing. */
    background: var(--bg); /* Follow the selected background color. */
    color: var(--fg); /* Follow the selected foreground color. */
    font: 16px/1.55 system-ui, sans-serif; /* Give prose a comfortable line height. */
} /* Finish the page defaults. */
main { max-width: 1000px; margin: 0 auto; padding: 32px 24px 64px; } /* Center the readable column. */
a { color: var(--accent); } /* Make links distinct in both themes. */
h1 { line-height: 1.1; } /* Keep the page title compact. */
h2 { margin-top: 44px; } /* Separate each API section. */
nav { display: flex; gap: 12px 20px; flex-wrap: wrap; margin: 24px 0; } /* Wrap section links. */
code { font: 0.92em ui-monospace, SFMono-Regular, Consolas, monospace; } /* Align source characters. */
pre { overflow: auto; padding: 16px; background: var(--code); border-radius: 8px; } /* Frame code blocks. */
pre code { white-space: pre-wrap; } /* Wrap long declarations on narrow screens. */
.entry { border-top: 1px solid var(--border); padding: 8px 0 18px; } /* Separate declaration cards. */
.entry p { margin: 10px 0; } /* Keep explanations close to their signatures. */
.source { font-size: 13px; } /* Keep source-line links smaller than prose. */
input { /* Make the search control match the page. */
    width: 100%; /* Span the documentation column. */
    padding: 12px; /* Keep the text easy to edit. */
    border: 1px solid var(--border); /* Show the input boundary. */
    border-radius: 6px; /* Match the code-panel corners. */
    background: var(--bg); /* Follow the selected theme. */
    color: var(--fg); /* Keep the search text readable. */
    font: inherit; /* Use the same type size as the prose. */
} /* Finish the search control styles. */
.muted { font-size: 14px; opacity: .78; } /* De-emphasize counts and generation notes. */
.hidden { display: none; } /* Hide entries excluded by the search. */
table { border-collapse: collapse; width: 100%; } /* Keep requirement columns aligned. */
th, td { padding: 8px; border-bottom: 1px solid var(--border); text-align: left; } /* Space table cells. */
.source-code code { white-space: pre; } /* Preserve exact header indentation. */
.line:target { background: var(--border); } /* Highlight a linked header line. */
.line-number { /* Keep source line numbers separate from the code. */
    display: inline-block; /* Reserve a consistent gutter. */
    width: 5em; /* Leave room for large header line numbers. */
    text-align: right; /* Align numbers by their final digit. */
    padding-right: 1.5em; /* Separate numbers from the source text. */
    opacity: .65; /* Keep the source itself more prominent. */
    text-decoration: none; /* Avoid underlining every line number. */
} /* Finish the source gutter styles. */
</style> <!-- All page styling is embedded above. -->
<link rel="stylesheet" href="site.css"> <!-- Match the navigation and appearance of the other documentation pages. -->
</head> <!-- The remaining page contains the API itself. -->
<body> <!-- Display the guide and its generated declaration cards. -->
<a class="skip-link" href="#content">Skip to content</a> <!-- Let keyboard readers skip the repeated navigation. -->
<header class="site-header"><div class="header-inner"><a class="brand" href="index.html">shdr<span class="release-version">__SHDR_DOC_VERSION__</span><span>CPU + GPU compute</span></a><nav class="site-nav" aria-label="Main navigation"><a href="index.html">Start</a><a href="examples.html">Examples</a><a href="results.html">Results</a><a href="corpus.html">Public shaders</a><a href="api.html" aria-current="page">API</a></nav></div></header> <!-- Connect the reference to the complete documentation site. -->
<div class="page-shell"><main class="page-main" id="content"> <!-- Keep the page within the readable content column. -->
<p class="eyebrow">Runtime reference</p> <!-- Distinguish declarations from the beginner walkthrough. -->
<h1>shdr __SHDR_DOC_VERSION__ API reference</h1> <!-- Name the library and release being documented. -->
<p>A C++11 header for CPU and GPU compute pipelines. The build helper turns native GLSL into CPU adapters and embeds GPU source. Applications create kernels, bind resources and run dispatches. <a href="source/shdr.h.html">Read or download the single-file runtime header</a>.</p> <!-- Connect the API to the runtime used by an application. -->
<p>Start with the <a href="index.html">walkthrough</a>. Read about <a href="resources.html">resources and bindings</a>, <a href="portability.html">supported shader features</a>, <a href="async.html">asynchronous execution</a>, <a href="imaging.html">imaging</a>, <a href="evt3.html">event-camera processing</a>, <a href="neural.html">neural models</a>, <a href="signals.html">signal processing</a> and <a href="datasets.html">public data</a>. The <a href="examples.html">example catalog</a> connects each complete program to its kernels.</p> <!-- Link the beginner guides. -->
<table> <!-- Compare the backend requirements. -->
<thead><tr><th scope="col">Backend</th><th scope="col">Minimum requirement</th></tr></thead> <!-- Label each requirement column. -->
<tbody> <!-- List the supported execution choices. -->
<tr><td>CPU</td><td>C++11 compiler and host resources.</td></tr> <!-- State the CPU requirement. -->
<tr><td>OpenGL ES</td><td>ES 3.1 with GLSL ES 3.10, or ES 3.2 with GLSL ES 3.20.</td></tr> <!-- State the GLES profiles. -->
<tr><td>Desktop OpenGL</td><td>GL 4.3 with GLSL 4.30, or a newer supported profile.</td></tr> <!-- State the desktop minimum. -->
</tbody> <!-- Finish the backend choices. -->
</table> <!-- Finish the requirement comparison. -->
<p>Enable <code>SHDR_ENABLE_EGL=1</code> and link EGL to create a headless GPU device. An existing application context can supply its own function resolver. GPU calls and final resource release use the context thread. The runtime handles transfers, dispatches, fence polling and bounded CPU work queues.</p> <!-- Explain context ownership and execution. -->
<nav class="section-nav" aria-label="API sections">"""
        + "\n".join(navigation)  # Insert the generated links to all nonempty API groups.
        + """</nav>
<label for="search">Find an API declaration</label><input id="search" type="search" placeholder="Try buffer, bind, glsl_source, pipeline, or imageLoad" autocomplete="off">
<p class="muted" id="count" aria-live="polite">"""
        + str(len(entries))  # Show the number of documented public entries.
        + """ documented declarations.</p>
"""
        + "\n".join(sections)  # Insert the rendered declaration sections in display order.
        + """
<details id="source"><summary>Browse the complete documented header source</summary><pre class="source-code"><code class="syntax-highlight">"""
        + "\n".join(  # Add the escaped header text to the source browser.
            f'<span class="line" id="L{line}"><a class="line-number" href="#L{line}">{line}</a>{text}</span>'  # Keep balanced token spans inside their numbered source line.
            for line, text in enumerate(highlight_lines(source, "cpp")[: len(source.splitlines())], 1)  # Tokenize the complete header so multiline comments retain their state.
        )  # Finish this part of the API page.
        + """</code></pre></details>
<script> // Keep search and source navigation available without external scripts.
const search = document.getElementById('search'); // Find the search input.
const entries = Array.from(document.querySelectorAll('.entry')); // Keep all declaration cards.
search.addEventListener('input', () => { // Refresh results after each text edit.
    const terms = search.value.toLowerCase().trim().split(/\\s+/).filter(Boolean); // Ignore case and empty words.
    let count = 0; // Count the entries that match every search word.
    entries.forEach(entry => { // Inspect each declaration's searchable text.
        const visible = terms.every(term => entry.dataset.search.includes(term)); // Require all words to match.
        entry.classList.toggle('hidden', !visible); // Hide unmatched declaration cards.
        if (visible) count++; // Count visible cards for the status text.
    }); // Every declaration now has its current visibility.
    document.querySelectorAll('section').forEach(section => { // Hide groups with no visible declarations.
        const visible = section.querySelector('.entry:not(.hidden)'); // Find any visible card in this group.
        section.classList.toggle('hidden', !visible); // Keep only groups with results.
    }); // Group headings now match their visible cards.
    document.getElementById('count').textContent = count + ' documented declarations match.'; // Show the result count.
}); // The input listener handles each new query.
function showSource() { // Open the header when a source-line link is selected.
    if (/^#L[0-9]+$/.test(location.hash)) { // Recognize only generated line anchors.
        document.getElementById('source').open = true; // Expand the source browser.
        const line = document.getElementById(location.hash.slice(1)); // Locate the linked line.
        if (line) line.scrollIntoView(); // Move the selected line into view.
    } // Ordinary section links leave the source browser unchanged.
} // Source navigation is also used for direct page links.
document.addEventListener('click', event => { // Open the source before scrolling to its anchor.
    if (event.target.closest('a[href^="#L"]')) { // Recognize a source-line link.
        document.getElementById('source').open = true; // Reveal the line's enclosing section.
    } // Other page links keep their usual behavior.
}); // Clicking a line link can now scroll to visible content.
window.addEventListener('hashchange', showSource); // Handle forward and back navigation.
showSource(); // Honor a source-line anchor in the initial page URL.
</script> <!-- Finish the embedded page behavior. -->
</main></div> <!-- Finish the documentation column. -->
<footer class="footer"><p>shdr __SHDR_DOC_VERSION__ · C++11 compute pipelines · <a href="portability.html">Portability and requirements</a> · <a href="source/index.html">Read the source</a></p></footer> <!-- Keep the next useful links available after the reference. -->
</body> <!-- Finish the visible page. -->
</html> <!-- The reference needs no network resources. -->
"""
    )  # Finish this part of the API page.
    document = document.replace("__SHDR_DOC_VERSION__", html.escape(version))  # Apply the header's release label to page chrome only.
    document = add_stylesheet(document, "highlight.css")  # Load the local stylesheet for precomputed syntax colors.
    output.parent.mkdir(parents=True, exist_ok=True)  # Create the output directory before writing the page.
    output.write_text(document, encoding="utf-8")  # Write the completed standalone API page.
    return len(entries)  # Report the number of exported entries.


def main() -> None:  # Read options and run this command-line tool.
    """Parse optional paths and export the library's API docstrings."""
    parser = argparse.ArgumentParser(description=__doc__)  # Describe the command and its supported options.
    parser.add_argument("--header", type=Path, default=ROOT / "shdr.h")  # Allow an alternate header while defaulting to this checkout's shdr.h.
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "api.html")  # Allow an alternate output while defaulting to docs/api.html.
    args = parser.parse_args()  # Read the header and output paths selected by the user.
    print(f"Exported {export(args.header, args.output)} documented declarations to {args.output}")  # Print the entry count and destination of the completed export.


if __name__ == "__main__":  # Run only when this file is invoked as a command.
    main()  # Read options and run the requested checks.
