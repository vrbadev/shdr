#!/usr/bin/env python3
"""Render static syntax highlighting with exact source text.

Source generators call highlight_code or highlight_lines before adding line
anchors. The command-line pass handles guide code excerpts and preserves logs,
tables and numbered sources. Pygments renders token spans during HTML creation.
Browsers display the resulting markup with the local stylesheet.
"""
import argparse  # Select HTML pages and an optional verification-only pass.
from html import escape, unescape  # Preserve code text while making token markup safe.
from html.parser import HTMLParser  # Locate the relevant code element boundaries.
import os  # Calculate a local stylesheet link for nested documentation pages.
from pathlib import Path  # Resolve source languages and static documentation files.
import re  # Recognize explicit language markers and conservative unlabeled examples.


ALIASES = {  # Accept the language names already used by the maintained guides.
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp", "c": "cpp",  # Use the established C++ lexer for the shared header and host examples.
    "glsl": "glsl", "vert": "glsl", "frag": "glsl", "comp": "glsl",  # Use native shader syntax for shader source bodies.
    "py": "python", "python3": "python", "python": "python",  # Keep Python examples independent of interpreter version spelling.
    "sh": "bash", "shell": "bash", "console": "bash", "bash": "bash",  # Read documented command lines as shell input.
    "cmake": "cmake", "js": "javascript", "javascript": "javascript",  # Support maintained build integration and browser helpers.
    "css": "css", "html": "html", "markdown": "markdown", "md": "markdown",  # Support the source browser's remaining text formats.
    "text": "text", "plaintext": "text", "plain": "text", "none": "text",  # Leave deliberate plain-text examples uncolored.
}  # Keep unknown language names on a safe plain-text fallback.


def language_for_path(path):  # Choose the source lexer from its path and language label.
    """Return a stable lexer name for a repository-relative source path."""
    path = Path(path)  # Accept both strings and Path objects from documentation generators.
    parts = path.as_posix().lower()  # Compare directory names consistently on all development hosts.
    if "/examples/shaders/" in "/" + parts or path.suffix.lower() in (".comp", ".vert", ".frag", ".glsl"):  # Use GLSL syntax for native shader .cpp files.
        return "glsl"  # Preserve native qualifiers, buffer declarations and builtin names.
    if path.name == "CMakeLists.txt" or path.suffix.lower() == ".cmake":  # Recognize both top-level and reusable CMake files.
        return "cmake"  # Use the established CMake command and variable rules.
    extensions = {".h": "cpp", ".hpp": "cpp", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".py": "python", ".sh": "bash", ".bash": "bash", ".js": "javascript", ".css": "css", ".html": "html", ".md": "markdown"}  # Cover every format selected by the source browser.
    return extensions.get(path.suffix.lower(), "text")  # Render other files as plain text.


def _pygments():  # Load the optional maintainer dependency only when highlighting is requested.
    try:  # Keep normal library builds and HTML reading independent of this package.
        from pygments.lexers import get_lexer_by_name  # Use the language lexer supplied by Pygments.
        from pygments.token import STANDARD_TYPES  # Reuse stable Pygments token category abbreviations.
    except ImportError as failure:  # Report exactly which optional documentation tool is missing.
        raise RuntimeError("Documentation highlighting requires Pygments. Install it with: python3 -m pip install Pygments. Reading the HTML and building shdr do not require it.") from failure  # Avoid a fallback that silently removes requested highlighting.
    return get_lexer_by_name, STANDARD_TYPES  # Keep the public helper import itself free of optional dependencies.


def _escape_text(text):  # Preserve carriage returns even when a browser normalizes HTML source newlines.
    return escape(text, quote=False).replace("\r", "&#13;")  # Preserve quote characters as source text.


def highlight_code(text, language="cpp"):  # Highlight a complete source before any numbered line wrapping.
    """Return escaped inline token spans with exact text.

    Full-input lexing preserves multiline comment and string state. Each line
    contains balanced spans and retains its original newline boundary. Callers
    add code wrappers, line numbers and navigation links around this markup.
    """
    lexer_factory, standard = _pygments()  # Give every caller the same optional dependency check.
    language = ALIASES.get(language.lower(), "text")  # Keep explicit plain text safe and predictable.
    lexer = lexer_factory(language, stripnl=False, stripall=False, ensurenl=False, tabsize=0)  # Disable whitespace trimming, added newlines and tab expansion.
    tokens = list(lexer.get_tokens_unprocessed(text))  # Avoid the convenience API's newline and byte-order-mark preprocessing.
    if "".join(value for _, _, value in tokens) != text:  # Refuse any lexer result that would change the displayed source.
        raise ValueError("Pygments changed source text while highlighting " + language)  # Keep exact source preservation an enforced contract.
    pieces = []  # Accumulate safe inline markup in original token order.
    for _, token, value in tokens:  # Reuse full-file lexer state across line boundaries.
        parent = token  # Find the nearest token category supported by the local stylesheet.
        while parent not in standard and parent.parent is not None:  # Handle lexer-specific token subcategories using established parent categories.
            parent = parent.parent  # Keep classification stable across Pygments versions.
        category = standard.get(parent, "")  # Append ordinary text directly.
        fragments = value.split("\n")  # Close token spans before every source line boundary.
        for index, fragment in enumerate(fragments):  # Reopen multiline token categories independently on later lines.
            if index:  # Emit exactly the newline present in the original token.
                pieces.append("\n")  # Keep line anchors aligned with the unchanged source text.
            if fragment:  # Select nonempty text fragments for token spans.
                encoded = _escape_text(fragment)  # Keep source markup and ampersands inert.
                pieces.append('<span class="tok-' + category + '">' + encoded + '</span>' if category else encoded)  # Apply local token style classes.
    return "".join(pieces)  # Return inline markup ready for existing source and signature wrappers.


def highlight_lines(text, language="cpp"):  # Share exact full-file lexing with numbered source generators.
    """Return balanced inline HTML lines, including a final empty line when present."""
    return highlight_code(text, language).split("\n")  # Let existing generators retain their own line-count and anchor conventions.


class _Stylesheets(HTMLParser):  # Read existing stylesheet links and their source positions.
    def __init__(self):  # Start with an empty set of declared stylesheet URLs.
        super().__init__(convert_charrefs=True)  # Decode HTML entities in existing href attributes.
        self.hrefs = set()  # Retain only links relevant to idempotent insertion.

    def handle_starttag(self, tag, attrs):  # Read the original link metadata.
        attrs = dict(attrs)  # Look up attributes independently of their ordering.
        if tag == "link" and "stylesheet" in attrs.get("rel", "").split():  # Ignore icons and unrelated resource links.
            self.hrefs.add(attrs.get("href", ""))  # Keep the exact already declared stylesheet target.


def add_stylesheet(document, href="highlight.css"):  # Add a local stylesheet while preserving page content and anchors.
    """Insert the given stylesheet before head closes, unless it is already linked."""
    existing = _Stylesheets()  # Preserve every unrelated byte in the original HTML.
    existing.feed(document)  # Inspect only existing link elements.
    if href in existing.hrefs:  # Reuse the existing generator-provided stylesheet.
        return document  # Keep repeated generation byte-idempotent.
    closing = re.search(r"</head\s*>", document, re.IGNORECASE)  # Find a complete document's stylesheet insertion point.
    if not closing:  # Avoid inserting presentation markup into a fragment or compiler log.
        raise ValueError("A complete HTML head is required to add highlighting CSS")  # Explain the caller's missing document shell.
    link = '<link rel="stylesheet" href="' + escape(href, quote=True) + '">'  # Keep the generated URL local and safely quoted.
    return document[:closing.start()] + link + document[closing.start():]  # Preserve all existing IDs, links and formatting.


class _CodeBlocks(HTMLParser):  # Locate eligible guide code while preserving unrelated HTML byte-for-byte.
    def __init__(self, document):  # Build offset mapping for the original document.
        super().__init__(convert_charrefs=False)  # Keep original entity spellings outside selected code interiors.
        self.document, self.blocks, self.pre, self.code = document, [], [], None  # Track only candidate pre/code sections.
        self.offsets = [0]  # Map HTMLParser line positions back to original string offsets.
        for line in document.splitlines(keepends=True):  # Preserve newline lengths while building the map.
            self.offsets.append(self.offsets[-1] + len(line))  # Retain the start offset for every parser line.

    def position(self):  # Convert the parser's current token position to an absolute offset.
        line, column = self.getpos()  # Read the beginning of the current HTML tag.
        return self.offsets[line - 1] + column  # Keep edits confined to the selected code interior.

    def handle_starttag(self, tag, attrs):  # Find code blocks and record any nested markup that must be preserved.
        attrs = dict(attrs)  # Read language and source-browser markers by name.
        if tag == "pre":  # Remember whether an existing numbered source wrapper owns this content.
            self.pre.append(attrs)  # Keep source blocks out of the guide postprocessor.
        if self.code is not None:  # Preserve existing anchors, links and syntax spans.
            self.code["nested"] = True  # Let the original generator own any structured code block.
        if tag == "code" and self.pre and self.code is None:  # Select explicit pre/code source excerpts.
            if "source-code" in self.pre[-1].get("class", "").split():  # Numbered files are highlighted before line wrapping by their generator.
                return  # Preserve structured source lines and encoded downloads.
            start = self.position() + len(self.get_starttag_text())  # Begin immediately after the unchanged code opening tag.
            self.code = {"start": start, "attrs": attrs, "nested": False}  # Store only the information needed for a safe interior replacement.

    def handle_endtag(self, tag):  # Close this candidate block at its original page boundary.
        if tag == "code" and self.code is not None:  # Capture the original unmodified interior boundaries.
            self.code["end"] = self.position()  # Stop before the original closing tag.
            self.blocks.append(self.code)  # Defer replacement until all original offsets are known.
            self.code = None  # Wait for the next explicit pre/code excerpt.
        if tag == "pre" and self.pre:  # Leave the current code or log wrapper.
            self.pre.pop()  # Preserve nesting information for later blocks.


def _block_language(attrs, text):  # Prefer explicit guide markers over conservative content recognition.
    for value in attrs.get("class", "").split():  # Accept the language-* and lang-* conventions used by static guides.
        if value.startswith(("language-", "lang-")):  # Keep explicit plain text out of automatic coloring.
            return ALIASES.get(value.split("-", 1)[1].lower(), "text")  # Keep the existing markup when its convention needs a custom parser.
    if re.search(r"\blayout\s*\(|\bgl_(?:Global|Local)InvocationID\b", text) or re.search(r"^\s*#version\b", text, re.MULTILINE):  # Recognize unlabeled native shader examples.
        return "glsl"  # Preserve shader qualifiers and builtins.
    if re.search(r"^\s*(?:cmake|ctest|python3?|git|SHDR_\w+=|\./build/|pip\s)\b", text, re.MULTILINE):  # Recognize the documented shell command prefixes.
        return "bash"  # Color the command text in its declared shell language.
    if re.search(r"\b(?:cmake_minimum_required|add_executable|target_link_libraries|shdr_target_shaders)\s*\(", text):  # Recognize unlabeled CMake integration excerpts.
        return "cmake"  # Use CMake variable and command rules.
    if re.search(r"^\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+\s*\()",
                 text, re.MULTILINE):  # Recognize Python source snippets.
        return "python"  # Keep imports and function definitions in their actual language.
    if re.search(r"\b(?:shdr::|std::|template\s*<|void\s+main\s*\(|#include|float\s+\w+\s*[=(])", text):  # Recognize the remaining documented C++ examples and signatures.
        return "cpp"  # Use the same lexer as the API declaration generator.
    return "text"  # Leave unclassified logs, diagrams and literal data unchanged.


def highlight_document(document, stylesheet="highlight.css"):  # Process guide excerpts and preserve structured sources and logs.
    """Return idempotently highlighted HTML with unchanged visible code and links."""
    reader = _CodeBlocks(document)  # Keep original offsets while inspecting block structure.
    reader.feed(document)  # Locate only explicit pre/code excerpts.
    for block in reversed(reader.blocks):  # Preserve earlier offsets while replacing later code interiors.
        if block["nested"]:  # Existing spans and anchors belong to a generator or earlier highlighting pass.
            continue  # Avoid duplicate token spans and line-gutter corruption.
        text = unescape(document[block["start"]:block["end"]])  # Recover the original visible code once before lexing.
        language = _block_language(block["attrs"], text)  # Respect explicit language markers and conservative fallback rules.
        if language == "text":  # Preserve raw logs and deliberate text examples byte-for-byte.
            continue  # Keep tables and diagram-like content out of syntax processing.
        encoded = highlight_code(text, language)  # Preserve full-block multiline state and all original code characters.
        document = document[:block["start"]] + encoded + document[block["end"]:]  # Retain the original code tags, IDs and surrounding prose.
    return add_stylesheet(document, stylesheet)  # Keep all highlighting local and independent of browser JavaScript.


def main():  # Apply the reusable guide postprocessor to a static documentation tree.
    root = Path(__file__).resolve().parent.parent  # Resolve default paths independently of the current shell directory.
    parser = argparse.ArgumentParser(description=__doc__)  # Explain that this is optional maintainer tooling.
    parser.add_argument("paths", nargs="*", type=Path)  # Accept individual HTML pages or a complete documentation tree.
    parser.add_argument("--docs", "--site", dest="docs", type=Path, default=root / "docs")  # Locate the shared local highlighting stylesheet.
    parser.add_argument("--check", action="store_true")  # Report stale highlighting in check mode.
    args = parser.parse_args()  # Read explicit regeneration or verification choices.
    files = sorted({file for path in (args.paths or [args.docs]) for file in (path.rglob("*.html") if path.is_dir() else [path]) if file.suffix.lower() == ".html"})  # Process only static HTML artifacts.
    changed = []  # Track files whose excerpt markup or stylesheet needs updating.
    try:  # Turn a missing optional package into one useful maintainer diagnostic.
        for path in files:  # Keep scientific SVGs and all non-HTML data outside this pass.
            before = path.read_text(encoding="utf-8")  # Read exact original markup for idempotence comparison.
            href = Path(os.path.relpath(args.docs / "highlight.css", path.parent)).as_posix()  # Keep nested source and benchmark styles local.
            after = highlight_document(before, href)  # Preserve logs, record values, IDs, anchors and downloads.
            if before != after:  # Avoid rewriting pages whose markup is already current.
                changed.append(path)  # Include the page in the maintainer summary.
                if not args.check:  # Verification mode must remain read-only.
                    path.write_text(after, encoding="utf-8")  # Write only the static presentation changes.
    except (RuntimeError, ValueError) as failure:  # Report missing dependencies or a preservation contract violation clearly.
        parser.exit(2, str(failure) + "\n")  # Avoid an opaque traceback for a routine documentation setup problem.
    print(("Needs highlighting: " if args.check else "Updated highlighting: ") + str(len(changed)) + " of " + str(len(files)) + " HTML pages")  # Summarize the updated page count.
    if args.check and changed:  # Let a maintainer or CI check detect stale presentation.
        parser.exit(1)  # Keep the command's status consistent with its verification result.


if __name__ == "__main__":  # Run the postprocessor only when invoked explicitly.
    main()  # Render or check the documentation's static syntax highlighting.
