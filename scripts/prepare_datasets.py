#!/usr/bin/env python3
"""Prepare public-data fixtures from checked source bytes.

Use --fetch to download missing recorded inputs. Use --verify to check local
prepared files with the Python standard library. Use --full-mnist to unpack
the complete official test split. JPEG preparation uses Pillow. Runtime
examples read the prepared binary files.
"""

import argparse  # Read preparation and verification options.
import gzip  # Unpack the original MNIST IDX files.
import hashlib  # Check source files and prepared fixture bytes.
from pathlib import Path  # Locate cache files and prepared outputs.
import struct  # Read IDX headers and write little-endian floating-point samples.
import urllib.request  # Download the recorded public input bytes.
import data_sources  # Share pinned downloads, prepared fingerprints and training provenance.

ROOT = Path(__file__).resolve().parents[1]  # Locate this source checkout.
METADATA = Path(data_sources.__file__).resolve()  # Keep maintained provenance in one importable Python module.


def sha256(data):  # Identify the exact bytes used by an example.
    return hashlib.sha256(data).hexdigest()  # Return the stable hexadecimal fingerprint.


def require(condition, message):  # Stop before accepting a malformed or changed fixture.
    if not condition:  # Check the format or checksum precondition.
        raise RuntimeError(message)  # Explain which source or transformation failed.


def source_bytes(entry, cache, fetch):  # Read one pinned input and download it only when requested.
    path = cache / entry["name"]  # Locate the cached source by its recorded filename.
    if not path.exists():  # Keep ordinary preparation free of network requests.
        require(fetch, "missing source " + entry["name"] + ". Run with --fetch")  # Require explicit download mode.
        headers = {"User-Agent": "shdr-public-data-preparation/1.0"}  # Identify this small public-data client.
        if "range" in entry:  # Limit large ECG and event recordings to an exact prefix.
            headers["Range"] = entry["range"]  # Ask for only the recorded byte interval.
        request = urllib.request.Request(entry["url"], headers=headers)  # Use the public endpoint and bounded request.
        with urllib.request.urlopen(request, timeout=60) as response:  # Close the connection after reading this input.
            if "range" in entry:  # Reject servers that ignore the bounded prefix request.
                require(response.status == 206, "server ignored byte range for " + entry["name"])  # Require a partial response.
            data = response.read(entry["bytes"] + 1)  # Read one extra byte to detect a changed source size.
        require(len(data) == entry["bytes"], "source size changed: " + entry["name"])  # Check the recorded response length.
        require(sha256(data) == entry["sha256"], "source checksum changed: " + entry["name"])  # Check bytes before caching them.
        path.write_bytes(data)  # Save only a complete verified source input.
    data = path.read_bytes()  # Read the source from the reusable cache.
    require(len(data) == entry["bytes"], "cached source size differs: " + entry["name"])  # Reject incomplete cached files.
    require(sha256(data) == entry["sha256"], "cached source checksum differs: " + entry["name"])  # Reject changed cache content.
    return data  # Return the verified source bytes to the conversion functions.


def mnist_images(data, count):  # Read the fixed MNIST image layout directly.
    unpacked = gzip.decompress(data)  # Expand the original compressed IDX file.
    require(len(unpacked) >= 16, "MNIST image header is incomplete")  # Require four big-endian header words.
    magic, actual_count, rows, columns = struct.unpack(">4I", unpacked[:16])  # Read type, count and image dimensions.
    require((magic, actual_count, rows, columns) == (2051, count, 28, 28), "unexpected MNIST image dimensions")  # Check the official split.
    require(len(unpacked) == 16 + count * 784, "MNIST image payload is incomplete")  # Require one byte per image pixel.
    return unpacked[16:]  # Extract raw row-major image bytes after the IDX header.


def mnist_labels(data, count):  # Read target classes in the original image order.
    unpacked = gzip.decompress(data)  # Expand the original compressed label file.
    require(len(unpacked) >= 8, "MNIST label header is incomplete")  # Require both big-endian header words.
    require(struct.unpack(">2I", unpacked[:8]) == (2049, count), "unexpected MNIST label count")  # Check the official split.
    labels = unpacked[8:]  # Keep one target byte per image.
    require(len(labels) == count and all(label < 10 for label in labels), "invalid MNIST label payload")  # Require digit classes zero through nine.
    return labels  # Preserve the original target order.


def prepare(output, full_output, inputs, full_mnist):  # Convert recorded inputs into small runtime fixtures.
    from PIL import Image, __version__ as pillow_version  # Decode the one JPEG during preparation only.

    artifacts = []  # Record every output file with its format and exact fingerprint.
    output.mkdir(parents=True, exist_ok=True)  # Create the destination for small committed fixtures.

    def save(name, data, description):  # Write one derived file and record how it was made.
        (output / name).write_bytes(data)  # Store the runtime-ready bytes.
        artifacts.append({  # Retain size, checksum and transformation beside each output.
            "name": name,  # Identify the prepared fixture by its relative filename.
            "bytes": len(data),  # Record the exact byte count for runtime readers.
            "sha256": sha256(data),  # Identify the final input used for numerical checks.
            **description,  # Add the source, format and transformation description.
        })  # Complete this fixture's provenance record.

    with Image.open(inputs["nasa-blue-marble.jpg"]) as original:  # Decode the checked NASA JPEG from its cache path.
        image = original.convert("RGB")  # Keep three unsigned color channels in RGB order.
    width, height = image.size  # Record the decoded source dimensions.
    require((width, height) == (800, 800), "unexpected NASA image dimensions")  # Check the pinned download's image size.
    rgb = image.tobytes()  # Read decoded pixels in row-major interleaved order.
    reduced = bytearray()  # Accumulate the sixty-four-by-sixty-four RGB fixture.
    for y in range(64):  # Visit every destination image row.
        source_y = (2 * y + 1) * height // 128  # Sample the center of this destination pixel in source coordinates.
        for x in range(64):  # Visit every destination image column.
            source_x = (2 * x + 1) * width // 128  # Use the same center-nearest rule horizontally.
            offset = 3 * (source_y * width + source_x)  # Locate the selected source RGB pixel.
            reduced.extend(rgb[offset:offset + 3])  # Copy the three decoded RGB channel bytes.
    save("earth_rgb.bin", bytes(reduced), {  # Store the small satellite-image fixture.
        "format": "RGB uint8, row-major, 64 by 64, three interleaved channels",  # Describe the byte layout.
        "source": "nasa-blue-marble.jpg",  # Link this output to its checked source entry.
        "transform": "JPEG decode to RGB, center-nearest downsample from 800 by 800 to 64 by 64",  # State the exact conversion.
        "pillow_version": pillow_version,  # Record the image decoder used to create these bytes.
    })  # Finish the image transformation record.

    header = inputs["mitdb-100.hea"].read_text()  # Read the ECG calibration accompanying the checked signal bytes.
    require(header.splitlines()[0] == "100 2 360 650000", "unexpected ECG record metadata")  # Check record, channels, rate and original sample count.
    channel_lines = header.splitlines()[1:3]  # Read the two channel-specific calibration lines.
    require(all(line.split()[1:5] == ["212", "200", "11", "1024"] for line in channel_lines), "unexpected ECG calibration")  # Check packed format, gain, resolution and zero level.
    encoded = inputs["mitdb-100-prefix.dat"].read_bytes()  # Read exactly 2048 pairs of packed samples.
    channels = [[], []]  # Collect both calibrated signals separately for channel-major storage.
    for offset in range(0, len(encoded), 3):  # Decode one pair of twelve-bit signed samples per three bytes.
        first, middle, last = encoded[offset:offset + 3]  # Read the shared-high-bits WFDB 212 encoding.
        values = (first | ((middle & 15) << 8), last | ((middle >> 4) << 8))  # Restore both twelve-bit ADC values.
        for channel, value in enumerate(values):  # Calibrate each of the two lead samples independently.
            if value >= 2048:  # Sign-extend a twelve-bit negative sample.
                value -= 4096  # Convert its unsigned bit pattern to the signed ADC value.
            channels[channel].append((value - 1024) / 200.0)  # Convert ADC counts into millivolts from the record header.
    samples = channels[0] + channels[1]  # Place all MLII samples before all V5 samples.
    save("signal_ecg.bin", struct.pack("<4096f", *samples), {  # Store 2048 samples per channel as little-endian floats.
        "format": "float32 little-endian, two channel-major arrays of 2048 samples",  # Describe the runtime signal layout.
        "source": "mitdb-100-prefix.dat",  # Link the waveform to its exact checked source interval.
        "transform": "WFDB 212 decode, first 2048 sample pairs, (ADC - 1024) / 200 millivolts",  # Record calibration and sample selection.
        "sample_rate_hz": 360,  # Preserve the official recording rate.
        "channels": ["MLII", "V5"],  # Preserve the lead names from the header.
    })  # Finish the signal transformation record.

    recording = inputs["laser-prefix.raw"].read_bytes()  # Read the bounded official EVT3 recording prefix.
    offset = 0  # Locate the first binary word after the ASCII metadata lines.
    while recording[offset:offset + 1] == b"%":  # Read the original RAW metadata lines before the event payload.
        offset = recording.index(b"\n", offset) + 1  # Advance to the next header line or the binary payload.
    require(b"% evt 3.0\n" in recording[:offset], "recording is not EVT3")  # Check the format identified by the source header.
    words = recording[offset:offset + 65536]  # Keep the exact first 32768 sixteen-bit words.
    require(len(words) == 65536, "EVT3 prefix does not contain enough complete words")  # Avoid an incomplete runtime fixture.
    save("evt3_words.bin", words, {  # Store the original encoded words with their timestamps and coordinates.
        "format": "uint16 little-endian, 32768 EVT3 words",  # Describe the decoder input.
        "source": "laser-prefix.raw",  # Link the fixture to its official checked recording prefix.
        "source_byte_offset": offset,  # Record where its original binary payload begins.
        "transform": "Remove ASCII RAW header, retain first 32768 complete encoded words",  # State that event bytes remain unchanged.
    })  # Finish the event transformation record.
    save("evt3_header.txt", recording[:offset], {  # Preserve the original recording metadata separately.
        "format": "ASCII RAW header",  # Describe the retained metadata encoding.
        "source": "laser-prefix.raw",  # Link this header to the same recording prefix.
        "transform": "Copy the exact header preceding binary words",  # Explain the separation of metadata and data.
    })  # Finish the original-header record.

    train_images = mnist_images(inputs["train-images-idx3-ubyte.gz"].read_bytes(), 60000)  # Read the complete official training images.
    train_labels = mnist_labels(inputs["train-labels-idx1-ubyte.gz"].read_bytes(), 60000)  # Read their matching digit targets.
    indices = []  # Keep the original indices selected for the small binary-classification fixture.
    class_counts = [0, 0]  # Select exactly sixty-four samples of each requested class.
    for index, label in enumerate(train_labels):  # Preserve original training order when selecting samples.
        if label < 2 and class_counts[label] < 64:  # Accept the first sixty-four instances of digit zero or one.
            indices.append(index)  # Record the source index before extracting its image.
            class_counts[label] += 1  # Update the quota for this digit.
        if class_counts == [64, 64]:  # Stop once the balanced subset is complete.
            break  # Keep later training examples outside this small fixture.
    save("mnist_train.bin", b"".join(train_images[index * 784:(index + 1) * 784] for index in indices), {  # Extract only the selected raw images.
        "format": "uint8, 128 row-major 28 by 28 images",  # Describe the training fixture dimensions.
        "source": "train-images-idx3-ubyte.gz",  # Identify the official training split.
        "transform": "First 64 instances each of digit 0 and 1 in original training order",  # Explain the balanced selection.
        "source_indices": indices,  # Make every selected training sample traceable.
    })  # Finish the binary training-image record.
    save("mnist_train_labels.bin", bytes(train_labels[index] for index in indices), {  # Extract labels in exactly the selected image order.
        "format": "uint8, 128 digit labels",  # Describe the corresponding target array.
        "source": "train-labels-idx1-ubyte.gz",  # Identify the original target split.
        "transform": "Labels at the indices recorded in mnist_train.bin",  # Preserve the selected image-to-target relationship.
    })  # Finish the binary training-label record.

    test_images = mnist_images(inputs["t10k-images-idx3-ubyte.gz"].read_bytes(), 10000)  # Read the untouched official test split.
    test_labels = mnist_labels(inputs["t10k-labels-idx1-ubyte.gz"].read_bytes(), 10000)  # Read its matching targets.
    save("mnist_test.bin", test_images[:512 * 784], {  # Ship a small reproducible prefix for ordinary example checks.
        "format": "uint8, 512 row-major 28 by 28 images",  # Describe the ten-class test fixture.
        "source": "t10k-images-idx3-ubyte.gz",  # Keep training and testing sources distinct.
        "transform": "Remove IDX header, retain first 512 official test images",  # Record the fixed test selection.
    })  # Finish the test-image record.
    save("mnist_labels.bin", test_labels[:512], {  # Retain the targets corresponding to the same test prefix.
        "format": "uint8, 512 digit labels",  # Describe the target array for the shipped test images.
        "source": "t10k-labels-idx1-ubyte.gz",  # Identify the official test-label file.
        "transform": "Remove IDX header, retain first 512 official test labels",  # Preserve original test ordering.
    })  # Finish the test-label record.
    if full_mnist:  # Also prepare the larger official split when explicitly requested.
        full_output.mkdir(parents=True, exist_ok=True)  # Keep full test data outside the small committed fixtures.
        (full_output / "mnist_full.bin").write_bytes(test_images)  # Write all ten thousand original test images.
        (full_output / "mnist_full_labels.bin").write_bytes(test_labels)  # Write all ten thousand matching target bytes.
    model_files = ("lenet_weights.bin", "lenet_reference.bin")  # Identify the separately trained binary model bundle.
    if any((output / name).exists() for name in model_files):  # Include model integrity when the bundle accompanies these fixtures.
        require(all((output / name).exists() for name in model_files), "LeNet model bundle is incomplete")  # Require weights and reference logits together.
        training = data_sources.TRAINING_PROVENANCE  # Read the independent training export from the maintained metadata module.
        weights = (output / "lenet_weights.bin").read_bytes()  # Read the original trained flat parameter tensor.
        logits = (output / "lenet_reference.bin").read_bytes()  # Read the framework's held-out reference scores.
        require(len(weights) == 61706 * 4 and sha256(weights) == training["weights_sha256"], "LeNet weights differ from training export")  # Check the exact expected parameter bundle.
        require(len(logits) == 10000 * 10 * 4 and sha256(logits) == training["reference_sha256"], "LeNet reference differs from training export")  # Check every held-out reference logit.
        formats = (  # Describe the two separately prepared binary model files.
            "float32 little-endian, 61706 LeNet parameters",  # Store convolution and dense tensors in documented flat order.
            "float32 little-endian, 10000 by 10 official-test reference logits",  # Preserve original test ordering for each class score.
        )  # Keep formats in the same order as the model filenames.
        for name, format_name in zip(model_files, formats):  # Record the existing trained model outputs.
            data = (output / name).read_bytes()  # Fingerprint the existing exported file.
            artifacts.append({  # Add this model artifact to the same offline integrity check.
                "name": name,  # Identify its relative filename in the fixture directory.
                "bytes": len(data),  # Record the exact exported file size.
                "sha256": sha256(data),  # Detect any change before native evaluation.
                "format": format_name,  # Describe how the C++ reader interprets its contents.
                "source": "scripts/train_lenet.py",  # Link to the independent reproducible training workflow.
                "transform": "Offline full-MNIST training and reference export described in lenet_training.json",  # Preserve the original record whose provenance now lives in data_sources.py.
            })  # Complete this model's integrity and provenance entry.
    return artifacts  # Return the exact physical-file provenance for the maintained metadata module.


def verify(output, artifacts=None):  # Check local data files and archived provenance.
    records = data_sources.PREPARED_ARTIFACTS if artifacts is None else artifacts  # Use maintained pins or the files just prepared by this process.
    for entry in records:  # Check each small prepared input independently.
        data = (output / entry["name"]).read_bytes()  # Read the fixture expected by runtime examples.
        require(len(data) == entry["bytes"], "fixture size differs: " + entry["name"])  # Detect truncation or replacement.
        require(sha256(data) == entry["sha256"], "fixture checksum differs: " + entry["name"])  # Detect any byte change.
    data_sources.verify_archived_training()  # Retain the original eleventh fingerprint after migrating its record into Python.
    print("Verified " + str(len(records)) + " data files + archived training-provenance fingerprint")  # Distinguish physical runtime files from the retained historical record.


def main():  # Select source preparation or offline fixture verification.
    parser = argparse.ArgumentParser(description=__doc__)  # Describe the optional dataset workflow.
    parser.add_argument("--fetch", action="store_true")  # Download missing sources only when this flag is supplied.
    parser.add_argument("--verify", action="store_true")  # Verify prepared inputs using the standard library.
    parser.add_argument("--full-mnist", action="store_true")  # Also unpack the complete ten-thousand-image test split.
    parser.add_argument("--cache", type=Path, default=ROOT / "build" / "datasets" / "upstream")  # Reuse downloaded source files.
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "data")  # Keep small fixtures with the example programs.
    parser.add_argument("--full-output", type=Path, default=ROOT / "build" / "datasets")  # Keep optional full test files in the build cache.
    parser.add_argument("--metadata", type=Path, default=METADATA)  # Update current fingerprints in this maintained Python metadata module.
    options = parser.parse_args()  # Read all paths and explicit preparation modes.
    try:  # Report preparation failures with a concise message.
        if options.verify:  # Use the lightweight offline integrity check when requested.
            verify(options.output)  # Compare every fixture to its recorded checksum.
            return  # Skip source preparation after successful verification.
        records = data_sources.SOURCE_RECORDS  # Read the pinned URLs, byte intervals and attribution from one maintained module.
        options.cache.mkdir(parents=True, exist_ok=True)  # Create the reusable source cache when needed.
        inputs = {}  # Collect paths only after source bytes have been verified.
        for entry in records:  # Check every input used by the deterministic preparation steps.
            source_bytes(entry, options.cache, options.fetch)  # Verify an existing source or download a missing one.
            inputs[entry["name"]] = options.cache / entry["name"]  # Retain its checked path for the conversion functions.
        artifacts = prepare(options.output, options.full_output, inputs, options.full_mnist)  # Produce small inputs and optional full test data.
        data_sources.write_metadata(artifacts, path=options.metadata)  # Refresh the current Python metadata records.
        verify(options.output, artifacts)  # Check the prepared files before reporting success.
    except (OSError, RuntimeError, ValueError, ImportError) as error:  # Handle missing inputs, formats and optional preparation dependencies.
        parser.exit(1, str(error) + "\n")  # Exit with a concise reason that identifies the failed preparation step.


if __name__ == "__main__":  # Run this workflow only when invoked as a command.
    main()  # Prepare or verify the requested public-data inputs.
