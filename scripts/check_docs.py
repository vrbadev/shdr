#!/usr/bin/env python3
"""Check the static documentation directory.

This check follows local links, checks fragment targets and verifies document
structure using the Python standard library. Browser review covers visual
presentation and interactive accessibility.
"""

import argparse  # Read an optional documentation directory.
from html.parser import HTMLParser  # Inspect links and document structure with the standard library.
from pathlib import Path  # Resolve the published files and their relative links.
from urllib.parse import unquote, urlsplit  # Separate local paths, fragments and external URLs.


ROOT = Path(__file__).resolve().parents[1]  # Locate the source checkout containing this tool.


class Page(HTMLParser):  # Collect the parts needed for structural and link checks.
    def __init__(self, path):  # Start a fresh parser for one published page.
        super().__init__(convert_charrefs=True)  # Decode HTML entities in attribute values.
        self.path = path  # Retain the file for helpful error messages.
        self.ids = set()  # Record valid fragment destinations on this page.
        self.links = []  # Record links and embedded assets with their line numbers.
        self.labels = []  # Record ARIA references to elements on the same page.
        self.tags = set()  # Track the structural elements present in the document.
        self.errors = []  # Collect failures across the complete site.
        self.has_language = False  # Require an explicit document language.
        self.has_viewport = False  # Require responsive sizing on narrow screens.

    def handle_starttag(self, tag, attributes):  # Inspect an opening tag and its attributes.
        attributes = dict(attributes)  # Make optional attribute lookups clear.
        line = self.getpos()[0]  # Retain the source line for diagnostics.
        self.tags.add(tag)  # Remember this document element.
        identifier = attributes.get("id")  # Read any fragment destination.
        if identifier:  # Check named destinations when a tag has an ID.
            if identifier in self.ids:  # Reject ambiguous fragment destinations.
                self.errors.append((line, "duplicate id " + identifier))  # Explain the conflicting destination.
            self.ids.add(identifier)  # Retain the destination for incoming links.
        if tag == "html":  # Read the language from the document root.
            self.has_language = bool(attributes.get("lang"))  # Require a nonempty language tag.
        if tag == "meta" and attributes.get("name", "").lower() == "viewport":  # Recognize responsive layout metadata.
            self.has_viewport = True  # Mark this page as ready for mobile sizing.
        if tag == "img" and "alt" not in attributes:  # Require text alternatives for all image elements.
            self.errors.append((line, "image has no alt attribute"))  # Empty alt remains valid for decorative images.
        for attribute in ("href", "src", "poster"):  # Follow both navigation links and displayed assets.
            if attributes.get(attribute):  # Ignore absent and empty optional URLs.
                self.links.append((line, attributes[attribute]))  # Retain the exact URL to check later.
        for attribute in ("aria-labelledby", "aria-describedby"):  # Check references used by assistive technology.
            for label in attributes.get(attribute, "").split():  # One attribute may name several local IDs.
                self.labels.append((line, label))  # Retain each referenced label.

    handle_startendtag = handle_starttag  # Inspect self-closing elements in the same way.


def check(site):  # Validate the complete site and return all helpful diagnostics.
    pages = {}  # Parse every page before resolving links between them.
    errors = []  # Keep failures from all files together.
    for path in sorted(site.rglob("*.html")):  # Visit every published HTML page in a stable order.
        page = Page(path)  # Create an independent parser for this document.
        page.feed(path.read_text(encoding="utf-8"))  # Inspect UTF-8 content with the standard HTML parser.
        page.close()  # Finish parsing any remaining text.
        pages[path.resolve()] = page  # Allow later fragment checks to find this page.
        for tag in ("html", "head", "title", "body", "main", "h1"):  # Require a usable page title and main content landmark.
            if tag not in page.tags:  # Identify a missing structural element.
                page.errors.append((1, "missing " + tag + " element"))  # Describe the missing page structure.
        if not page.has_language:  # Ensure screen readers can choose a document language.
            page.errors.append((1, "missing document language"))  # Explain the missing language metadata.
        if not page.has_viewport:  # Ensure the responsive stylesheet can size the page correctly.
            page.errors.append((1, "missing viewport metadata"))  # Explain the missing mobile metadata.
    for path, page in pages.items():  # Resolve links after all fragment destinations are known.
        for line, label in page.labels:  # Check each accessibility label reference.
            if label not in page.ids:  # Require the label to exist on this page.
                page.errors.append((line, "missing ARIA label target " + label))  # Explain the broken accessibility reference.
        for line, url in page.links:  # Inspect every navigation link and local asset.
            parts = urlsplit(url)  # Separate the path from a fragment or query.
            if parts.scheme or parts.netloc:  # External references are outside this offline check.
                if parts.scheme.lower() == "javascript":  # Require ordinary navigation URLs.
                    page.errors.append((line, "javascript URL in document navigation"))  # Request a normal link or button.
                continue  # Keep external links as recorded references.
            if parts.path.startswith("/"):  # Root paths fail when Pages serves a project below its repository name.
                page.errors.append((line, "root-relative URL " + url))  # Identify links that escape the published directory.
                continue  # Avoid interpreting the URL as an operating-system path.
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path  # Resolve a local relative destination.
            if target.is_dir():  # Match ordinary static-server directory navigation.
                target = target / "index.html"  # Resolve the directory's entry document.
            if target != site and site not in target.parents:  # Keep all published dependencies inside docs.
                page.errors.append((line, "URL leaves the published directory: " + url))  # Explain the missing Pages dependency.
            elif not target.is_file():  # Require every local page and asset to be present.
                page.errors.append((line, "missing local target " + url))  # Identify the broken link or asset.
            elif parts.fragment and target in pages:  # Check fragment IDs in linked HTML documents.
                fragment = unquote(parts.fragment)  # Decode any URL-escaped fragment characters.
                if fragment not in pages[target].ids:  # Require a matching destination ID.
                    page.errors.append((line, "missing fragment " + url))  # Identify the broken section link.
        errors.extend((path, line, message) for line, message in page.errors)  # Keep the filename beside each diagnostic.
    if not (site / "index.html").is_file():  # Require the entry file expected by GitHub Pages.
        errors.append((site, 1, "missing index.html entry page"))  # Require a landing page at the site root.
    return pages, errors  # Return parsed pages for a summary and all detected problems.


def main():  # Expose the validation as a small command-line tool.
    parser = argparse.ArgumentParser(description=__doc__)  # Explain the offline validation behavior.
    parser.add_argument("--site", type=Path, default=ROOT / "docs")  # Allow checking a copied site before deployment.
    options = parser.parse_args()  # Read the selected published directory.
    site = options.site.resolve()  # Compare canonical local paths consistently.
    pages, errors = check(site)  # Read the complete site and report structural findings.
    for path, line, message in errors:  # Print all failures in an editor-friendly format.
        print("{}:{}: {}".format(path.relative_to(site) if site in path.parents else path, line, message))  # Show one precise problem per line.
    if errors:  # Report failure to CMake or a future continuous-integration job.
        raise SystemExit(1)  # Keep unsuccessful publication checks distinct from a valid site.
    print("Checked {} HTML pages and their local links and assets".format(len(pages)))  # Summarize the completed structural check.


if __name__ == "__main__":  # Run only when invoked as a command.
    main()  # Validate the requested documentation directory.
