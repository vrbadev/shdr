#!/usr/bin/env python3
"""Render complete public-shader compiler records as an accessible standalone report.

The renderer accepts the verifier's ordinary Python report dictionary. Every
recorded leaf value remains visible and carries its original field path, so a
reader can inspect every result directly in the report.
"""
from html import escape  # Keep source names and diagnostics inert inside HTML.
from pathlib import Path  # Read the documented library version from the runtime header.
import re  # Turn recorded field names into readable headings.


def documentation_version():  # Identify the library version and preserve recorded report fields.
    header = Path(__file__).resolve().parent.parent / "shdr.h"  # Locate the runtime documented by this checkout.
    match = re.search(r'^\s*#define\s+SHDR_VERSION\s+"([^"]+)"', header.read_text(encoding="utf-8"), re.MULTILINE)  # Read the header's public version string.
    if not match:  # Require an identifiable library version in the report title.
        raise ValueError("SHDR_VERSION is missing from shdr.h")  # Identify the missing version declaration.
    return escape(match.group(1))  # Escape the library version for HTML.


STYLE = """
body{margin:0;background:#f5f7fb;color:#17243a;font:16px/1.6 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:2rem 1.4rem}a{color:#155a9c}
h1,h2,h3{line-height:1.2}h1{font-size:clamp(2rem,5vw,3.5rem)}h2{margin-top:2.8rem}
.lead{max-width:78ch;color:#46546a;font-size:1.12rem}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-weight:700;color:#52657d}
.report-nav{display:flex;gap:1rem;flex-wrap:wrap}.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}
.card,.callout{background:white;border:1px solid #d6deea;border-radius:12px;padding:1.1rem}.card strong{font-size:2rem;display:block}
.callout{border-left:4px solid #1c7790}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{padding:.65rem .75rem;text-align:left;vertical-align:top;border-bottom:1px solid #d6deea}th{font-weight:700}
caption{text-align:left;font-weight:700;margin-bottom:.5rem}code{font: .89em ui-monospace,monospace;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f7;padding:1rem;border-radius:6px;font: .85rem/1.55 ui-monospace,monospace}
details{background:white;border:1px solid #d6deea;border-radius:8px;padding:.8rem 1rem;margin:.7rem 0}
summary{cursor:pointer;font-weight:650}summary:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,button:focus-visible{outline:3px solid #ce6a2e;outline-offset:3px}
.record-path{overflow-wrap:anywhere}.status{display:inline-block;font-size:.78rem;border-radius:20px;background:#e4ebf5;padding:.15rem .55rem;margin-left:.5rem}
.status.compiled{background:#d9f0e6;color:#165b43}.controls{display:flex;gap:.8rem;align-items:end;flex-wrap:wrap;margin:1rem 0}
.controls label{display:flex;flex-direction:column;gap:.3rem}.controls input,.controls select,.controls button{font:inherit;padding:.55rem;border:1px solid #8c9bb0;border-radius:5px}
.controls input{min-width:min(25rem,75vw)}.field-table th{width:15rem}section.record-section{margin:1.2rem 0}.record-section h4{margin-bottom:.3rem}
.skip-link{position:absolute;left:-9999px}.skip-link:focus{left:1rem;top:.3rem;background:white;padding:.5rem;z-index:2}
[hidden]{display:none!important}.footer{max-width:1160px;margin:2rem auto;padding:1rem 1.4rem;border-top:1px solid #d6deea;color:#52657d}
@media print{body{background:white}.controls,.site-nav{display:none}details{break-inside:avoid}a{color:inherit}}
"""  # Supply all report styles when the output is opened outside the documentation site.


SEARCH = """
(() => {
  const records = [...document.querySelectorAll('.corpus-record')];
  const text = records.map(record => record.textContent.toLowerCase());
  const query = document.getElementById('record-search');
  const repository = document.getElementById('repository-filter');
  const status = document.getElementById('status-filter');
  const count = document.getElementById('record-count');
  function filter() {
    const words = query.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);
    let visible = 0;
    records.forEach((record, index) => {
      const show = (!repository.value || record.dataset.repository === repository.value)
        && (!status.value || record.dataset.status === status.value)
        && words.every(word => text[index].includes(word));
      record.hidden = !show;
      if (show) visible += 1;
    });
    count.textContent = `${visible} of ${records.length} source records shown`;
  }
  query.addEventListener('input', filter);
  repository.addEventListener('change', filter);
  status.addEventListener('change', filter);
  document.getElementById('expand-records').addEventListener('click', () => {
    records.filter(record => !record.hidden).forEach(record => { record.open = true; });
  });
  document.getElementById('collapse-records').addEventListener('click', () => {
    records.forEach(record => { record.open = false; });
  });
  filter();
})();
"""  # Search the rendered records directly in the browser.


def label(value):  # Make recorded field names readable while retaining their exact spellings below.
    return re.sub(r"[_-]+", " ", str(value)).capitalize()  # Separate words in report identifiers.


def scalar(value, path):  # Render one exact report value with a machine-checkable original path.
    kind = "null" if value is None else "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "string"  # Preserve the original scalar category.
    text = "null" if value is None else "true" if value is True else "false" if value is False else str(value)  # Format booleans and numeric values according to their recorded types.
    attrs = f'data-report-path="{escape(path, quote=True)}" data-report-type="{kind}"'  # Identify this value independently of its presentation.
    if text.startswith(("https://", "http://")) and "\n" not in text:  # Keep recorded source and license URLs directly usable.
        return f'<a {attrs} href="{escape(text, quote=True)}">{escape(text)}</a>'  # Show the complete source URL and its provenance.
    tag = "pre" if "\n" in text or path.endswith(".diagnostic") else "span"  # Keep diagnostic line breaks available to readers and preservation checks.
    return f'<{tag} {attrs}>{escape(text)}</{tag}>'  # Escape compiler output before placing it in the document.


def fields(value, path):  # Render nested report dictionaries and lists with every field retained.
    if isinstance(value, dict):  # Expose every dictionary entry with its original key.
        rows = []  # Build readable key and value rows in source order.
        for key, child in value.items():  # Preserve optional configuration and compiler fields as well as the usual ones.
            rendered = fields(child, path + "." + key)  # Keep a complete path for every nested value.
            rows.append(f'<tr><th scope="row"><code>{escape(key)}</code></th><td>{rendered}</td></tr>')  # Show exact field names alongside their values.
        return '<div class="table-wrap"><table class="field-table"><tbody>' + "".join(rows) + "</tbody></table></div>"  # Keep wide hashes and diagnostics usable on small screens.
    if isinstance(value, list):  # Preserve source paths and any future list-valued metadata.
        return "<ol>" + "".join(f"<li>{fields(child, path + '[' + str(index) + ']')}</li>" for index, child in enumerate(value)) + "</ol>"  # Retain list order in the visible document.
    return scalar(value, path)  # Render the final scalar verbatim.


def navigation(site_navigation):  # Choose sibling-page navigation when the report is part of the guide.
    if not site_navigation:  # Keep arbitrary verifier output paths fully standalone.
        return '<nav class="report-nav" aria-label="Report sections"><a href="#summary">Summary</a><a href="#workflow">Import a shader</a><a href="#provenance">Provenance</a><a href="#records">All source records</a></nav>'  # Use only anchors present in this report.
    return '<header class="site-header"><div class="header-inner"><a class="brand" href="index.html">shdr<span class="release-version">' + documentation_version() + '</span><span>CPU + GPU compute</span></a><nav class="site-nav" aria-label="Main navigation"><a href="index.html">Start</a><a href="examples.html">Examples</a><a href="results.html">Results</a><a href="corpus.html" aria-current="page">Public shaders</a><a href="api.html">API</a></nav></div></header>'  # Match the shared static documentation shell.


def import_guide(site_navigation):  # Explain imports through the documented language subset.
    subset = '<a href="portability.html">supported language subset</a>' if site_navigation else "supported language subset described by this library"  # Avoid broken sibling links in standalone verifier output.
    return f"""
<section id="workflow"><h2>Import a public compute shader</h2>
<p>Start with a compute shader that reads and writes independent buffer elements. Add its source to <code>shdr_target_shaders</code>, bind its declared resources by name, and run the resulting pipeline on the CPU or a compatible GPU. An existing <code>.comp</code> extension is accepted. The build embeds the portable GPU variant directly in the application.</p>
<p>Use a unique identifier-shaped source basename in each linked target, such as <code>multiply.comp</code> or <code>filter.cpp</code>. The CMake helper derives its factory name from that basename. Each basename identifies one factory within the target.</p>
<p>Backend compatibility depends on buffer layout, builtins, extensions and synchronization. Consult the {subset} before importing shared workgroup memory, atomics, samplers or custom records.</p>
<p>A mutable global with an explicit initializer starts from that value for every invocation on both backends. Each invocation receives independent private global storage. Put shared or persistent data in a bound buffer or image, and assign uninitialized private variables before reading them.</p></section>
<section id="scope"><h2>What these checks establish</h2>
<p>The selected inputs cover complete compute programs, compiler conformance fixtures and support units. Reference acceptance means that the original input compiled and linked in OpenGL mode, or compiled to Vulkan SPIR-V when its declarations required Vulkan. CPU translation covers inputs accepted by at least one reference mode. Every record retains both reference outcomes.</p>
<p>Checked-in reference transcripts identify upstream fixtures with expected compiler diagnostics. This metadata and the standalone compiler result describe their respective test configurations. Every source record preserves the original expectation flag.</p>
<div class="callout"><strong>Compiler coverage.</strong> Successful exports establish compiler compatibility with the five language profiles. Applications supply the host resources, dimensions and valid input domains for runtime execution. Bundled runtime examples measure CPU/GPU agreement against independent references on the listed devices.</div>
<p>Devices must also support the imported resource counts, workgroup sizes and input dimensions. The bundled runtime examples separately compare CPU and GPU values, including independent numerical references. Inspect each record and validate an imported kernel with its intended inputs before deployment.</p></section>
<section id="configuration"><h2>Source assembly and portable imports</h2>
<p>NCNN sources depend on host-supplied types and load/store macros. The verifier inserts the exact pinned <code>src/ncnn_glsl_ext.comp</code> prelude and records the FP32 buffer configuration. This configuration uses FP32 buffer storage. The records retain the shared-memory paths selected by <code>NCNN_shader_local_memory=1</code> and their compiler diagnostics. Upstream activation includes are resolved by glslang before CPU translation. The upstream body remains unchanged. Configuration, prelude, configured-source and preprocessed-source hashes are shown in the affected records.</p>
<p>The adapter selects a version, assigns buffer bindings and layouts, maps descriptor set zero and specialization defaults, and exposes supported push-constant members as named uniforms. The portable GPU variant supplies numeric promotions required by GLSL ES, removes local <code>const</code> qualifiers from runtime initializers, and handles integer boolean-selector <code>mix</code> on older profiles. It preserves expression evaluation order and keeps output and input-output arguments as lvalues. Global and compile-time local constants retain their qualifiers. These imports use the OpenGL resource and dispatch API.</p>
<p>The dual-backend subset covers independent invocations, single-array storage blocks, fixed local arrays, documented storage images and math operations. The detailed records identify requirements involving workgroup coordination, atomics, multi-field storage blocks, additional image formats, samplers, subgroups, cooperative matrices and reduced-precision types through their compiler diagnostics.</p></section>
<section id="reproduce"><h2>Reproduce the compiler checks</h2>
<p>Use Python 3.8 or newer, Git, a C++11 compiler and <code>glslangValidator</code>. The commands below fetch pinned inputs explicitly into a separate cache.</p>
<pre><code># Fetch the pinned collection into a separate cache and write a standalone report.
python3 scripts/verify_public_shaders.py --fetch \\
  --cache build/public-shaders --output build/public-shader-results.html
# Reuse the prepared local cache.
python3 scripts/verify_public_shaders.py \\
  --cache build/public-shaders --output build/public-shader-results.html</code></pre>
<p>A build configured with glslang also provides <code>cmake --build build --target verify_public_shaders</code>. Populate its cache first. The <code>--jobs</code>, <code>--cxx</code> and <code>--glslang</code> options select parallelism and tools. Each report records fingerprints for the implementation held constant throughout verification.</p>
<p>Fetch the upstream sources with the command above and follow their licenses when incorporating them. Every repository pin, source link and license link is retained below.</p></section>
"""  # Keep the complete import, scope, assembly and reproduction explanation with the report itself.


def render_public_report(report, site_navigation=False):  # Render the complete report with the Python standard library.
    version = documentation_version()  # Show the library version alongside the recorded compiler fingerprints.
    counts = report["summary"]["counts"]  # Use measured totals to describe compiler coverage.
    records = report["results"]  # Preserve every selected source, including reference rejections.
    compiled = counts.get("cpu_and_all_profiles_compiled", 0)  # Count only sources accepted by C++ and every generated profile.
    accepted = counts.get("opengl_passed", 0) + counts.get("vulkan_only_passed", 0)  # Keep OpenGL acceptance distinct from the Vulkan fallback.
    extra_css = '<link rel="stylesheet" href="site.css">' if site_navigation else ""  # Use shared site styling only where that asset is available.
    pieces = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Complete pinned public compute-shader compiler records with source provenance and diagnostics."><title>Public shader compatibility · shdr {version}</title><style>{STYLE}</style>{extra_css}</head><body><a class="skip-link" href="#content">Skip to content</a>{navigation(site_navigation)}<div class="page-shell"><main class="page-main" id="content">']  # Make the output readable with JavaScript disabled or outside GitHub Pages.
    pieces.append(f'<p class="eyebrow">Recorded compiler compatibility</p><h1>Every source, every outcome</h1><p class="lead">Explore {len(records)} public compute inputs from five pinned collections. Read the original links, exact configurations, compiler outcomes and diagnostics.</p><div class="card-grid"><div class="card"><strong>{len(records)}</strong>selected inputs</div><div class="card"><strong>{accepted}</strong>reference accepted</div><div class="card"><strong>{compiled}</strong>C++11 and all five profiles</div><div class="card"><strong>0</strong>public corpus shaders executed</div></div>')  # State compiler coverage and its execution limit together.
    pieces.append('<section id="summary"><h2>Recorded totals</h2>' + fields(report["summary"], "$.summary") + "</section>")  # Retain all aggregate counts and rejection categories.
    pieces.append(import_guide(site_navigation))  # Explain how to interpret and reproduce the outcomes.
    metadata = {key: value for key, value in report.items() if key not in ("results", "summary", "repositories")}  # Keep every top-level provenance field exactly once.
    pieces.append('<section id="provenance"><h2>Compiler and implementation provenance</h2>' + fields(metadata, "$") + '<p>The recorded header, frontend and tool fingerprints identify the implementation used for these compiler checks. Source records retain the original compiler diagnostics.</p></section>')  # Preserve tool versions, hashes, selection rules and excluded-input counts.
    pieces.append('<section id="repositories"><h2>Pinned repositories and licenses</h2>')  # Keep source selection and licensing together in the report.
    for index, repository in enumerate(report["repositories"]):  # Retain the original order of the five recorded collections.
        pieces.append('<details open><summary>' + escape(repository["id"]) + "</summary>" + fields(repository, f"$.repositories[{index}]") + "</details>")  # Include every pin, path, assembly link and optional configuration field.
    pieces.append('</section><section id="records"><h2>All source records</h2><p>Search paths, hashes, compiler messages or configuration values. Repository and CPU-status filters combine with the search. Every record uses native expandable HTML. Search and filters use local JavaScript.</p><div class="controls"><label>Search records<input id="record-search" type="search" placeholder="Path, builtin, hash or diagnostic"></label><label>Repository<select id="repository-filter"><option value="">All repositories</option>')  # Use native accessible controls over fully rendered content.
    for repository in report["repositories"]:  # Offer only repository choices represented by this report.
        key = escape(repository["id"], quote=True)  # Escape even verifier-controlled identifiers before embedding attributes.
        pieces.append(f'<option value="{key}">{key}</option>')  # Keep filter values identical to recorded repository IDs.
    pieces.append('</select></label><label>CPU result<select id="status-filter"><option value="">All outcomes</option>')  # Keep not-attempted records separate from compiler failures.
    for status in sorted({record["cpu"]["status"] for record in records}):  # Derive the available CPU outcomes from the recorded results.
        pieces.append(f'<option value="{escape(status, quote=True)}">{escape(label(status))}</option>')  # Offer readable status labels while preserving exact values.
    pieces.append('</select></label><button id="expand-records" type="button">Expand shown</button><button id="collapse-records" type="button">Collapse all</button></div><p id="record-count" role="status" aria-live="polite">' + str(len(records)) + ' source records</p>')  # Update the result count while keyboard focus stays in place.
    for index, record in enumerate(records):  # Show every detailed source record in its original order.
        state = record["cpu"]["status"]  # Identify the CPU outcome in the closed summary.
        repository = record["repository"]  # Group search results by the original collection identifier.
        pieces.append(f'<details class="corpus-record" id="record-{index + 1:04d}" data-repository="{escape(repository, quote=True)}" data-status="{escape(state, quote=True)}"><summary><span class="record-path">{escape(repository)} / {escape(record["path"])}</span><span class="status {escape(state, quote=True)}">{escape(label(state))}</span></summary>{fields(record, f"$.results[{index}]")}</details>')  # Preserve source hashes, configured hashes and all compiler-stage diagnostics.
    pieces.append(f'</section></main></div><footer class="footer"><p>shdr {version} · Public collection compiler report. Resource limits and numerical behavior require application-specific validation.</p></footer><script>' + SEARCH + "</script></body></html>\n")  # Inline search logic supports standalone reading.
    return "\n".join(pieces)  # Return the complete standalone HTML report.
