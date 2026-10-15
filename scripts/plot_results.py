#!/usr/bin/env python3
"""Render SVG figures from the visible result tables and supplied fixtures.

NumPy and Matplotlib render the figures. The visible HTML tables provide the
recorded measurements, and prepared data files provide the example imagery.
"""
import argparse  # Select a published result page or an alternate output directory.
from html.parser import HTMLParser  # Read visible table cells with the standard-library HTML parser.
from pathlib import Path  # Locate the versioned reports, fixtures and SVG outputs.
import re  # Add accessible descriptions to the standalone vector figures.


class ReportTables(HTMLParser):  # Collect the explicit publication tables and exact corpus scalar fields.
    def __init__(self):  # Initialize the active table and captured field state.
        super().__init__(convert_charrefs=True)  # Decode visible HTML entities back into their recorded text.
        self.tables, self.values = {}, {}  # Retain table rows and path-addressed corpus values.
        self.active, self.row, self.cell = None, None, None  # Track only the table currently being parsed.
        self.row_attrs, self.capture = {}, None  # Retain optional row identity and one scalar capture.

    def handle_starttag(self, tag, attrs):  # Recognize semantic table boundaries and report-value annotations.
        attrs = dict(attrs)  # Access attributes by their names.
        if tag == "table" and attrs.get("id"):  # Select tables carrying an authoritative identifier.
            self.active = attrs["id"]  # Remember the current publication table.
            self.tables[self.active] = []  # Collect its rows in displayed order.
        if self.active and tag == "tr":  # Begin a row only within an identified table.
            self.row, self.row_attrs = [], attrs  # Preserve visible cells and any stable row identity.
        if self.row is not None and tag in ("td", "th"):  # Read both headings and ordinary data cells.
            self.cell = []  # Begin accumulating this cell's visible text.
        if attrs.get("data-report-path"):  # Read exact scalar provenance already visible in the corpus page.
            self.capture = (tag, attrs["data-report-path"], [])  # Keep the closing tag and original dictionary path.

    def handle_data(self, data):  # Preserve human-readable table text and recorded scalar values.
        if self.cell is not None:  # Include nested code and link text in the current table cell.
            self.cell.append(data)  # Keep text until the cell closes.
        if self.capture is not None:  # Preserve the exact corpus value separately from surrounding table headings.
            self.capture[2].append(data)  # Retain line breaks and scalar spelling.

    def handle_endtag(self, tag):  # Finish each collected value at its semantic boundary.
        if self.capture is not None and tag == self.capture[0]:  # Store a complete path-addressed report field.
            self.values[self.capture[1]] = "".join(self.capture[2])  # Avoid changing diagnostics or hash strings.
            self.capture = None  # Stop collecting after that scalar's closing tag.
        if tag in ("td", "th") and self.cell is not None:  # Finish one displayed table cell.
            self.row.append(" ".join("".join(self.cell).split()))  # Normalize only table-layout whitespace.
            self.cell = None  # Wait for the next cell.
        if tag == "tr" and self.row is not None:  # Finish one complete publication row.
            self.tables[self.active].append((self.row_attrs, self.row))  # Retain attributes and cells together.
            self.row = None  # Stop collecting cells until another row begins.
        if tag == "table":  # Leave the current identified table.
            self.active = None  # Collect text inside the selected table cells.


def read_tables(path):  # Parse one local standalone HTML report.
    reader = ReportTables()  # Use only the standard HTML parser to recover the visible record.
    reader.feed(path.read_text(encoding="utf-8"))  # Read the complete published page.
    return reader  # Return both visible tables and exact corpus metadata.


def numeric_rows(report, identifier, start):  # Skip headings and return only rows with numeric measured fields.
    result = []  # Keep the original visible row order.
    for attrs, cells in report.tables[identifier]:  # Visit each heading or data row.
        try:  # Parse measurements from numeric cells.
            values = [float(value) for value in cells[start:]]  # Read the exact rounded numbers published in the table.
        except ValueError:  # Keep headings out of the numerical record.
            continue  # Continue with the next displayed row.
        result.append((attrs, cells[:start], values))  # Retain labels and row identities with their numerical values.
    return result  # Use the recorded measurement values and timing ratios.


def save_svg(figure, output, title, description):  # Export a self-contained scientific vector figure.
    figure.savefig(output, format="svg", bbox_inches="tight", metadata={"Date": "2026-10-15", "Creator": "shdr publication plots", "Title": title})  # Preserve the recorded measurement date.
    text = output.read_text(encoding="utf-8")  # Add screen-reader text to the exported SVG root.
    from html import escape  # Keep plot descriptions safe inside XML.
    description = f'<title id="plot-title">{escape(title)}</title><desc id="plot-description">{escape(description)}</desc>'  # Explain units and scope inside the downloadable artifact.
    text = re.sub(r"(<svg\b[^>]*)(>)", r'\1 role="img" aria-labelledby="plot-title plot-description"\2' + description, text, count=1)  # Make the SVG usable independently of its HTML figure caption.
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"  # Remove exporter padding while preserving the SVG's paths and text.
    output.write_text(text, encoding="utf-8")  # Keep every plot as a portable standalone file.
    print(output.name)  # Report the artifact generated by this invocation.


def benchmark_plot(report, host, title, graphics, output, np, plt):  # Compare all 25 measured pipelines on one configuration.
    rows = numeric_rows(report, "benchmark-" + host, 2)  # Read the exact visible CPU, graphics and reported ratio columns.
    labels = [label[0] for _, label, _ in rows]  # Preserve names in the same order as the benchmark executable.
    values = np.asarray([value for _, _, value in rows])  # Keep recorded rounded means and unrounded-derived reported ratios.
    y = np.arange(len(rows))  # Assign one shared position to each pipeline.
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 12.2), gridspec_kw={"width_ratios": [2.6, 1]}, sharey=True)  # Align time and relative-completion views for direct reading.
    axes[0].barh(y - .17, values[:, 0], height=.32, color="#225d9b", label="C++11 CPU")  # Plot the measured serial CPU means.
    axes[0].barh(y + .17, values[:, 1], height=.32, color="#0c8a79", label=graphics)  # Distinguish llvmpipe software from actual hardware in the legend.
    axes[0].set_xscale("log")  # Keep microsecond-scale and large neural workloads visible together.
    axes[0].set_yticks(y, labels, fontsize=9)  # Make every pipeline label readable in the vector export.
    axes[0].invert_yaxis()  # Match the published table's top-to-bottom ordering.
    axes[0].set_xlabel("Mean completed workload time (ms, logarithmic axis)")  # State the measured unit and nonlinear axis clearly.
    axes[0].grid(axis="x", which="both", alpha=.2)  # Help compare bars across the broad timing range.
    axes[0].legend(loc="upper right", frameon=False)  # Place the execution-path legend beside the row labels.
    axes[1].axvline(1, color="#566071", linewidth=1, linestyle="--")  # Mark equal completion time explicitly.
    axes[1].scatter(values[:, 2], y, s=34, color="#bd5a2d", zorder=3)  # Use the benchmark's recorded CPU/GPU ratio.
    axes[1].set_xscale("log")  # Show advantages in either direction on a symmetric ratio scale.
    axes[1].set_xlim(.02, 40)  # Use the same ratio range for all three configurations.
    axes[1].set_xticks([.02, .1, 1, 10, 40], ["0.02×", "0.1×", "1×", "10×", "40×"])  # Avoid opaque scientific notation for the relative-completion view.
    axes[1].set_xlabel("CPU time / graphics time")  # Label the plotted value as the CPU/GPU time ratio.
    axes[1].grid(axis="x", which="both", alpha=.2)  # Retain readable ratio comparisons at small values.
    axes[1].tick_params(axis="y", left=False, labelleft=False)  # Keep a single shared pipeline-label column.
    figure.suptitle(title, fontsize=18, x=.02, ha="left")  # Name the tested CPU and graphics implementation.
    figure.text(.02, .025, "20 measured repetitions · seed 12345 · scale 1 · ES 3.1 profile\nResident means exclude compilation, first upload and final readback. Hybrid transfers remain included.", fontsize=10, color="#46546a")  # Keep timing scope attached to downloaded figures.
    figure.tight_layout(rect=(0, .06, 1, .97))  # Reserve room for the title and measurement note.
    save_svg(figure, output, title, "All 25 recorded pipeline timings and CPU-to-graphics ratios. Timing means are in milliseconds. Ratios greater than one favor the graphics path.")  # Export the scientific plot with accessible units and interpretation.
    plt.close(figure)  # Release figure memory before generating the next configuration.


def evaluation_plot(report, output, np, plt):  # Compare the six recorded complete held-out evaluations.
    rows = numeric_rows(report, "mnist-metrics", 2)  # Read correct counts, image counts, accuracy, seconds and maximum error.
    labels = [text[0] + " / " + text[1] for _, text, _ in rows]  # Keep each CPU and graphics configuration explicit.
    accuracy = np.array([values[2] for _, _, values in rows])  # Preserve the logged accuracy percentages.
    seconds = np.array([values[3] for _, _, values in rows])  # Keep full-evaluation scope separate from resident benchmark time.
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.3))  # Show accuracy and completed duration side by side.
    y = np.arange(len(rows))  # Assign the same row to each path in both panels.
    axes[0].barh(y, accuracy, color="#0c8a79", label="Correct")  # Show measured successful classifications on a full percentage scale.
    axes[0].barh(y, 100 - accuracy, left=accuracy, color="#d9a06f", label="Incorrect")  # Show the complete observed error fraction.
    axes[0].set_xlim(0, 100)  # Avoid exaggerating equal accuracies with a truncated axis.
    axes[0].set_yticks(y, labels, fontsize=9)  # Retain the identity of all six measured paths.
    axes[0].invert_yaxis()  # Match the accessible result table order.
    axes[0].set_xlabel("Official held-out labels (%)")  # State the denominator and unit.
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)  # Explain both correct and incorrect portions of every bar.
    axes[1].barh(y, seconds, color=["#225d9b" if index % 2 == 0 else "#0c8a79" for index in y])  # Distinguish CPU and graphics durations consistently.
    axes[1].set_yticks(y, labels, fontsize=9)  # Keep the duration panel usable on its own.
    axes[1].invert_yaxis()  # Align paths with the accuracy panel.
    axes[1].set_xlabel("Completed 10,000-image evaluation (seconds)")  # Include preprocessing and comparison in the stated scope.
    for index, value in enumerate(seconds):  # Label actual recorded durations directly on the figure.
        axes[1].text(value + .4, index, f"{value:g}", va="center", fontsize=9)  # Avoid estimating values from the axis.
    axes[1].set_xlim(0, max(seconds) * 1.18)  # Leave room for the duration annotations.
    axes[1].grid(axis="x", alpha=.2)  # Help compare completed evaluation times.
    figure.suptitle("Complete MNIST evaluation · 9,778 / 10,000 correct on every path", fontsize=16)  # State the exact observed classification result.
    figure.tight_layout(rect=(0, 0, 1, .94))  # Keep labels and title clear of each other.
    save_svg(figure, output, "Complete held-out MNIST evaluations", "Every recorded path classified 9778 of 10000 labels correctly, or 97.78 percent. Duration includes batch construction, input preparation, native execution, result reads and reference comparison. llvmpipe uses the CPU.")  # Keep model-evaluation scope with the plot.
    plt.close(figure)  # Release plotting resources.


def coverage_plot(report, output, np, plt):  # Show passed tests and skipped profiles as separate outcomes.
    rows = numeric_rows(report, "runtime-tests", 1)  # Read passed, skipped, failed and total counts.
    values = np.array([value for _, _, value in rows], dtype=int)  # Keep test outcomes as exact integer counts.
    labels = [text[0] for _, text, _ in rows]  # Label configurations by their public hardware names.
    figure, axis = plt.subplots(figsize=(11.5, 4.2))  # Give each registered configuration a clearly separated bar.
    y = np.arange(len(rows))  # Position the three platform totals.
    axis.barh(y, values[:, 0], color="#0c8a79", label="Passed")  # Count only tests that actually completed successfully.
    axis.barh(y, values[:, 1], left=values[:, 0], color="#c5cedb", label="Unavailable context")  # Show profile-availability skips explicitly.
    axis.barh(y, values[:, 2], left=values[:, 0] + values[:, 1], color="#c3443b", label="Failed")  # Keep a zero-valued failure category explicit in the legend.
    for index, row in enumerate(values):  # Label the plotted bars with exact counts.
        axis.text(row[3] + 1, index, f"{row[0]} passed · {row[1]} skipped · {row[2]} failed", va="center", fontsize=10)  # Include the absence of failures and the scope of skipped contexts.
    axis.set_yticks(y, labels)  # Identify the tested renderer configuration.
    axis.invert_yaxis()  # Preserve the order of the source table.
    axis.set_xlim(0, max(values[:, 3]) + 47)  # Reserve room for complete count labels.
    axis.set_xlabel("Registered CTest cases")  # Distinguish registered cases from individual shader/profile compilations.
    axis.legend(frameon=False, loc="lower right")  # Identify skips as profile-availability outcomes.
    axis.set_title("Runtime and integration validation · no failed registered tests", loc="left", fontsize=15)  # Describe observed outcomes with their unit of coverage.
    figure.tight_layout()  # Keep count labels inside the exported bounds.
    save_svg(figure, output, "Registered test outcomes", "Passed, skipped and failed registered CTest cases for the three configurations. Some registered cases contain many numerical checks. The separate 250 shader/profile checks and eight frontend tests are described in the result tables.")  # Prevent confusion between suite totals and inner assertion counts.
    plt.close(figure)  # Release the figure after export.


def loss_plot(report, output, np, plt):  # Plot the measured initial and final mini-batch losses.
    rows = numeric_rows(report, "mini-batch-loss", 1)  # Read step count, starting loss and final loss.
    labels = [text[0] for _, text, _ in rows]  # Keep the three small model names visible.
    values = np.array([value for _, _, value in rows])  # Use the identical logged values observed across all six paths.
    x = np.arange(len(rows))  # Separate the three architectures.
    figure, axis = plt.subplots(figsize=(9, 4.8))  # Use bars because only the start and finish were recorded.
    axis.bar(x - .18, values[:, 1], width=.35, color="#8ca7c5", label="Before updates")  # Show the observed starting cross entropy.
    axis.bar(x + .18, values[:, 2], width=.35, color="#0c8a79", label="After updates")  # Show the observed final mini-batch cross entropy.
    axis.set_xticks(x, [name + f"\n{int(value[0])} updates" for name, value in zip(labels, values)])  # Make differing update counts explicit.
    axis.set_ylim(0, .82)  # Show a true zero baseline for the loss comparison.
    axis.set_ylabel("Mean binary cross entropy")  # Use the models' actual scalar loss definition.
    axis.legend(frameon=False)  # Distinguish the two recorded observations.
    axis.set_title("Real digit mini-batch training · 32 selected zero/one images", loc="left", fontsize=14)  # Avoid calling training-set loss a held-out accuracy result.
    axis.grid(axis="y", alpha=.2)  # Help compare measured loss changes.
    figure.tight_layout()  # Keep model and update-count labels readable.
    save_svg(figure, output, "Observed real-data mini-batch loss changes", "Recorded loss before and after training for ANN, CNN and transformer models on one 32-image training mini-batch. No intermediate learning curve or generalization metric was recorded.")  # State the limited scope of these observations.
    plt.close(figure)  # Release plotting resources.


def confusion_plot(report, output, np, plt):  # Plot the complete independent-framework reference confusion table.
    rows = numeric_rows(report, "reference-confusion", 1)  # Read the ten visible ground-truth rows and predicted-class counts.
    matrix = np.array([value for _, _, value in rows], dtype=int)  # Preserve all 100 actual confusion counts.
    from matplotlib.colors import LogNorm  # Make rare confusions readable beside large correct-class totals.
    figure, axis = plt.subplots(figsize=(8.5, 7.3))  # Keep each digit pair legible in the standalone SVG.
    image = axis.imshow(np.ma.masked_equal(matrix, 0), cmap="Blues", norm=LogNorm(vmin=1, vmax=matrix.max()))  # Leave zero-count cells white and display positive counts logarithmically.
    for row in range(10):  # Annotate every original digit class.
        for column in range(10):  # Annotate every predicted digit class.
            value = matrix[row, column]  # Use the exact displayed reference count.
            axis.text(column, row, str(value), ha="center", va="center", fontsize=8, color="white" if value > 50 else "#17243a")  # Keep large and small counts readable against the color scale.
    axis.set_xticks(range(10))  # Label predictions with their original digit values.
    axis.set_yticks(range(10))  # Label true classes with the official MNIST labels.
    axis.set_xlabel("Predicted digit · independent framework reference")  # Avoid claiming that native per-class predictions were archived.
    axis.set_ylabel("Official test digit")  # State the ground-truth source.
    axis.set_title("Reference confusion · 10,000 official test images", loc="left", fontsize=14)  # Define the complete evaluated population.
    figure.colorbar(image, ax=axis, label="Reference predictions per cell (log scale)")  # State the nonlinear color unit.
    figure.tight_layout()  # Keep axis labels and colorbar visible.
    save_svg(figure, output, "Independent reference model confusion", "Exact confusion counts from the shipped independent framework logits and all 10000 official labels. Native per-class prediction matrices were not retained. Native complete scores and total accuracy were checked separately.")  # Keep the distinction between native and reference predictions explicit.
    plt.close(figure)  # Release plotting resources.


def fixture_plot(fixtures, output, np, plt):  # Illustrate actual prepared inputs and an explicitly independent signal reference.
    earth = np.frombuffer((fixtures / "earth_rgb.bin").read_bytes(), dtype=np.uint8).reshape(64, 64, 3)  # Read the exact prepared NASA RGB image.
    ecg = np.frombuffer((fixtures / "signal_ecg.bin").read_bytes(), dtype="<f4").reshape(2, 2048)[0].astype(float)  # Read the channel-major MLII signal in millivolts.
    digits = np.frombuffer((fixtures / "mnist_test.bin").read_bytes(), dtype=np.uint8).reshape(-1, 28, 28)  # Read the shipped official test-image prefix.
    labels = np.frombuffer((fixtures / "mnist_labels.bin").read_bytes(), dtype=np.uint8)  # Retain the original digit labels.
    indices = [int(np.flatnonzero(labels == value)[0]) for value in range(10)]  # Show the first supplied instance of each class.
    offsets = np.arange(65) - 32  # Center the documented odd-length FIR design.
    coefficients = 2 * (40 / 360) * np.sinc(2 * (40 / 360) * offsets) * np.hamming(65)  # Independently calculate the Hamming-sinc 40 Hz low-pass reference.
    coefficients = (coefficients / coefficients.sum()).astype(np.float32).astype(float)  # Match stored coefficient precision while keeping the reference accumulation independent.
    filtered = np.convolve(ecg, coefficients)[:len(ecg)]  # Apply the documented causal zero-prefix FIR convention.
    time = np.arange(len(ecg)) / 360  # Express sample positions in seconds at the public recording's sample rate.
    frequencies = np.fft.rfftfreq(len(ecg), 1 / 360)  # Express the real-spectrum bins in hertz.
    figure = plt.figure(figsize=(14.5, 8.5))  # Combine image, waveform, spectrum and digit examples into one clear fixture plate.
    grid = figure.add_gridspec(3, 3, height_ratios=[1, 1, .65], width_ratios=[1, 1.1, 1.1])  # Reserve a full-width final row for the ten digit classes.
    axis = figure.add_subplot(grid[:2, 0])  # Display the supplied low-resolution image.
    axis.imshow(earth, interpolation="nearest")  # Preserve the prepared RGB8 pixels.
    axis.set_title("NASA Blue Marble · prepared 64 × 64 RGB", fontsize=11)  # Identify the source and prepared image size.
    axis.axis("off")  # Avoid implying a geographic coordinate system for the RGB preview.
    axis = figure.add_subplot(grid[0, 1:])  # Show the original ECG and independent filter reference together.
    axis.plot(time, ecg, linewidth=.8, color="#7a8fa8", label="Prepared MLII")  # Plot the actual float32 waveform from the fixture.
    axis.plot(time, filtered, linewidth=1, color="#0c8a79", label="Independent 65-tap, 40 Hz FIR reference")  # Clearly distinguish a computed reference from archived runtime output.
    axis.set_xlim(0, 2.5)  # Show the original samples from an initial waveform interval.
    axis.set_xlabel("Time (s)")  # State waveform time units.
    axis.set_ylabel("Potential (mV)")  # Preserve the published ADC scaling.
    axis.legend(frameon=False, fontsize=9, ncol=2)  # Identify source samples and reference filtering.
    axis.grid(alpha=.2)  # Make waveform amplitude comparisons readable.
    axis = figure.add_subplot(grid[1, 1:])  # Show the independent full-fixture real Fourier spectrum.
    for signal, color, name in [(ecg, "#7a8fa8", "Prepared MLII"), (filtered, "#0c8a79", "Independent FIR reference")]:  # Use the same sources and colors as the waveform panel.
        amplitude = np.abs(np.fft.rfft(signal)) / len(signal)  # Compute a correctly normalized real-signal amplitude spectrum.
        amplitude[1:-1] *= 2  # Double only the interior positive-frequency bins.
        axis.semilogy(frequencies, np.maximum(amplitude, 1e-9), color=color, linewidth=.9, label=name)  # Use an explicit small display floor on the logarithmic amplitude axis.
    axis.set_xlabel("Frequency (Hz)")  # Use physical frequency units on the axis.
    axis.set_ylabel("Single-sided amplitude (mV)")  # State the spectrum normalization and signal unit.
    axis.set_xlim(0, 180)  # Show the original 360 Hz recording's complete Nyquist range.
    axis.grid(alpha=.2)  # Help locate the low-pass attenuation region.
    digit_grid = grid[2, :].subgridspec(1, 10)  # Keep one original labeled sample per digit class.
    for digit, index in enumerate(indices):  # Preserve an explicit reproducible sample-selection rule.
        axis = figure.add_subplot(digit_grid[0, digit])  # Give each test digit its own clean preview.
        axis.imshow(digits[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")  # Display the original unsigned-byte image values.
        axis.set_title(f"Label {digit}\nTest index {index}", fontsize=9)  # Distinguish ground-truth labels from model predictions.
        axis.axis("off")  # Keep pixel previews uncluttered.
    figure.suptitle("Recorded public inputs · images, ECG and official handwritten digits", fontsize=16)  # Describe the figure as actual input data and reference signal analysis.
    figure.tight_layout(rect=(0, 0, 1, .95))  # Separate the title from the specimen panels.
    save_svg(figure, output, "Actual prepared public fixtures", "Actual NASA image pixels, PhysioNet MLII samples at 360 Hz, independently calculated 40 Hz FIR and Fourier references, and first shipped MNIST instance of each class. This is an input illustration, not an invented runtime measurement.")  # Keep source and reference distinctions with the standalone artifact.
    plt.close(figure)  # Release plotting resources.


def main():  # Reproduce all figures from the published HTML and bundled data.
    root = Path(__file__).resolve().parent.parent  # Resolve paths relative to the project.
    parser = argparse.ArgumentParser(description=__doc__)  # Explain that plotting is an optional publication step.
    parser.add_argument("--report", type=Path, default=root / "docs/results.html")  # Read the authoritative visible measurement tables.
    parser.add_argument("--output", type=Path, default=root / "docs/plots")  # Choose where standalone SVGs are written.
    parser.add_argument("--fixtures", type=Path, default=root / "examples/data")  # Read the prepared public input bytes.
    args = parser.parse_args()  # Apply explicit caller paths before importing optional plotting packages.
    import numpy as np  # Calculate only reference illustrations and array-based plot coordinates.
    import matplotlib  # Keep scientific rendering outside normal library builds.
    matplotlib.use("Agg")  # Render figures with the headless plotting backend.
    import matplotlib.pyplot as plt  # Render standard scientific vector plots.
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "shdr-publication-2026-10-15", "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 12})  # Keep selectable text, clear axes and consistent publication typography.
    report = read_tables(args.report)  # Recover all plotted measurements from their accessible tables.
    args.output.mkdir(parents=True, exist_ok=True)  # Create only the requested publication artifact directory.
    for host, title, graphics in [("amd", "Ryzen 9 5900X · C++11 and llvmpipe software", "llvmpipe software"), ("intel", "Core i7-1260P · C++11 and Iris Xe", "Iris Xe hardware"), ("pi", "Raspberry Pi 5 · C++11 and V3D", "V3D hardware")]:  # Generate the same benchmark views for all measured configurations.
        benchmark_plot(report, host, title, graphics, args.output / ("benchmark-" + host + ".svg"), np, plt)  # Preserve platform identity and exact recorded numbers.
    evaluation_plot(report, args.output / "mnist-evaluation.svg", np, plt)  # Show all six complete held-out measurements.
    coverage_plot(report, args.output / "test-coverage.svg", np, plt)  # Separate successful tests from unavailable-context skips.
    loss_plot(report, args.output / "mini-batch-loss.svg", np, plt)  # Plot only the before-and-after training observations actually recorded.
    confusion_plot(report, args.output / "mnist-reference-confusion.svg", np, plt)  # Reproduce the full visible reference confusion matrix.
    fixture_plot(args.fixtures, args.output / "public-fixtures.svg", np, plt)  # Illustrate the real shipped input bytes with clearly labeled reference analysis.


if __name__ == "__main__":  # Run plotting only when requested explicitly.
    main()  # Generate the standalone publication figures.
