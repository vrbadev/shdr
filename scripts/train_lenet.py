#!/usr/bin/env python3
"""Train the documented modern LeNet variant and export plain portable weights.

PyTorch is an optional preparation tool. The C++ examples load only float arrays.
Training uses the official MNIST training split and evaluates the separate test split.
"""
import argparse  # Describe the preparation inputs and reproducible training settings.
import gzip  # Read the original compressed IDX input files.
import hashlib  # Record exact training inputs and exported model bytes.
from pathlib import Path  # Resolve cache and output paths with pathlib.
import struct  # Decode the big-endian IDX headers from the published dataset.
import time  # Record completed training time for the preparation report.
import data_sources  # Share pinned MNIST inputs and the maintained Python provenance module.


def digest(path):  # Identify a file by its complete SHA-256 digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()  # Hash the actual bytes used here.


def read_idx(path, dimensions):  # Validate and decode one official MNIST IDX archive.
    raw = gzip.decompress(path.read_bytes())  # Expand only the named dataset archive.
    magic = struct.unpack_from(">I", raw)[0]  # Read the IDX type and dimension tag.
    if magic != 0x0800 + dimensions:  # Accept only unsigned-byte images or labels.
        raise ValueError(f"unexpected IDX header in {path.name}")  # Reject another data format.
    shape = struct.unpack_from(">" + "I" * dimensions, raw, 4)  # Read every dimension.
    offset = 4 + 4 * dimensions  # Locate the first payload byte after the header.
    size = 1  # Begin the expected payload-size product.
    for dimension in shape:  # Include every declared array dimension.
        size *= dimension  # Count the unsigned-byte values in the archive.
    if len(raw) != offset + size:  # Check both truncation and trailing unexpected content.
        raise ValueError(f"IDX payload size differs in {path.name}")  # Report unsafe input.
    return raw[offset:], shape  # Return verified payload bytes and their shape.


def main():  # Keep optional training dependencies outside ordinary C++ builds.
    parser = argparse.ArgumentParser(description=__doc__)  # Explain the preparation command.
    parser.add_argument("--cache", type=Path, required=True)  # Locate downloaded official IDX files.
    parser.add_argument("--output", type=Path, default=Path("examples/data"))  # Choose fixture storage.
    parser.add_argument("--epochs", type=int, default=3)  # Use a small but complete training run.
    parser.add_argument("--batch-size", type=int, default=128)  # Control each optimizer batch.
    parser.add_argument("--seed", type=int, default=12345)  # Reproduce initialization and batch order.
    parser.add_argument("--threads", type=int, default=8)  # Bound preparation CPU parallelism.
    parser.add_argument("--metadata", type=Path, default=Path(data_sources.__file__).resolve())  # Store training results beside the maintained dataset records.
    args = parser.parse_args()  # Read the explicit training configuration.
    if min(args.epochs, args.batch_size, args.threads) <= 0:  # Reject empty or invalid training runs.
        parser.error("epochs, batch size and thread count must be positive")  # Explain the constraint.
    import numpy as np  # Export endian-defined arrays and decode verified dataset bytes.
    import torch  # Use the optional reference framework only during preparation.
    from torch import nn  # Construct the documented standard convolutional model.
    torch.set_num_threads(args.threads)  # Avoid using every host core for a small example.
    torch.manual_seed(args.seed)  # Fix both initial parameters and training permutation.
    torch.use_deterministic_algorithms(True)  # Select deterministic CPU preparation operations.

    class LeNet(nn.Module):  # Name the modern LeNet variant implemented by the native shaders.
        def __init__(self):  # Declare layers in exported parameter order.
            super().__init__()  # Initialize the framework's ordinary layer bookkeeping.
            self.conv1 = nn.Conv2d(1, 6, 5)  # Map padded input to six 28 by 28 feature maps.
            self.conv2 = nn.Conv2d(6, 16, 5)  # Map pooled features to sixteen 10 by 10 maps.
            self.dense1 = nn.Linear(400, 120)  # Apply the full C5-equivalent dense layer.
            self.dense2 = nn.Linear(120, 84)  # Apply the second hidden classifier layer.
            self.dense3 = nn.Linear(84, 10)  # Produce one logit per handwritten digit class.

        def forward(self, values):  # Match the native seven-dispatch inference pipeline.
            values = nn.functional.pad(values, (2, 2, 2, 2))  # Add a zero border to 32 by 32.
            values = nn.functional.avg_pool2d(torch.relu(self.conv1(values)), 2)  # First conv and pool.
            values = nn.functional.avg_pool2d(torch.relu(self.conv2(values)), 2)  # Second conv and pool.
            values = torch.relu(self.dense1(values.flatten(1)))  # Flatten channel-major pooled maps.
            values = torch.relu(self.dense2(values))  # Apply the second hidden ReLU layer.
            return self.dense3(values)  # Keep raw logits for stable cross entropy and comparison.

    names = [  # Preserve the exact official split filenames in provenance.
        "train-images-idx3-ubyte.gz",  # Read all 60,000 training images.
        "train-labels-idx1-ubyte.gz",  # Read only their training labels during optimization.
        "t10k-images-idx3-ubyte.gz",  # Keep the 10,000-image test split held out.
        "t10k-labels-idx1-ubyte.gz",  # Use test labels only for final evaluation.
    ]  # Keep the training split independent of evaluation samples.
    paths = [args.cache / name for name in names]  # Find each verified archive in the public cache.
    pinned = {entry["name"]: entry for entry in data_sources.SOURCE_RECORDS}  # Reuse the exact download pins used by preparation.
    for path in paths:  # Check every original image and label archive before training reads it.
        record = pinned[path.name]  # Find this official split's expected compressed bytes.
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:  # Reject changed inputs even when their IDX shape is plausible.
            raise ValueError("MNIST source fingerprint differs: " + path.name)  # Protect the recorded training and test provenance.
    train_bytes, train_shape = read_idx(paths[0], 3)  # Decode training images with validated shape.
    train_labels, label_shape = read_idx(paths[1], 1)  # Decode training class labels separately.
    test_bytes, test_shape = read_idx(paths[2], 3)  # Decode the held-out image split.
    test_labels, test_label_shape = read_idx(paths[3], 1)  # Decode the held-out class labels.
    if train_shape != (60000, 28, 28) or test_shape != (10000, 28, 28):  # Require official dimensions.
        raise ValueError("the expected official MNIST split sizes are required")  # Reject reshaped inputs.
    if label_shape != (60000,) or test_label_shape != (10000,):  # Pair every image with its own label.
        raise ValueError("MNIST image and label counts differ")  # Reject incomplete training inputs.
    images = torch.from_numpy(np.frombuffer(train_bytes, dtype=np.uint8).copy())  # Own training bytes.
    images = images.reshape(60000, 1, 28, 28).float().div_(255.0)  # Match native zero-to-one scaling.
    labels = torch.from_numpy(np.frombuffer(train_labels, dtype=np.uint8).copy()).long()  # Class IDs.
    test = torch.from_numpy(np.frombuffer(test_bytes, dtype=np.uint8).copy())  # Own test image bytes.
    test = test.reshape(10000, 1, 28, 28).float().div_(255.0)  # Apply identical test preprocessing.
    targets = torch.from_numpy(np.frombuffer(test_labels, dtype=np.uint8).copy()).long()  # Test classes.
    model = LeNet()  # Create the same layer shapes and parameter order as the C++ helper.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Learn all model weights and biases.
    start = time.perf_counter()  # Start timing after data preparation.
    history = []  # Record each complete training epoch's mean cross entropy.
    for epoch in range(args.epochs):  # Visit the complete training split once per epoch.
        model.train()  # Make the preparation intent explicit for future layer changes.
        order = torch.randperm(images.shape[0])  # Shuffle only the training examples.
        loss_sum = 0.0  # Accumulate sample-weighted training loss for this epoch.
        for offset in range(0, len(order), args.batch_size):  # Include the final partial batch.
            indices = order[offset:offset + args.batch_size]  # Select this batch's original samples.
            optimizer.zero_grad(set_to_none=True)  # Release the previous batch's gradient values.
            logits = model(images[indices])  # Evaluate the complete deep network.
            loss = nn.functional.cross_entropy(logits, labels[indices])  # Compute supervised loss.
            loss.backward()  # Differentiate every layer in the reference preparation framework.
            optimizer.step()  # Update all trainable weights from the current batch.
            loss_sum += float(loss.detach()) * len(indices)  # Weight partial batches correctly.
        history.append(loss_sum / len(order))  # Record completed sample-mean epoch loss.
        print(f"epoch {epoch + 1}: training cross entropy {history[-1]:.6f}", flush=True)  # Show progress.
    training_seconds = time.perf_counter() - start  # Record completed training duration.
    model.eval()  # Freeze preparation behavior before held-out evaluation.
    with torch.no_grad():  # Preserve model parameters during evaluation.
        logits = torch.cat([model(batch) for batch in test.split(256)])  # Bound evaluation memory.
        accuracy = float((logits.argmax(1) == targets).float().mean())  # Measure all 10,000 labels.
        test_loss = float(nn.functional.cross_entropy(logits, targets))  # Record held-out loss too.
    weights = np.concatenate([p.detach().numpy().reshape(-1) for p in model.parameters()])  # Export order.
    if weights.size != 61706:  # Catch an accidental architecture or parameter-layout change.
        raise ValueError("LeNet parameter layout must contain exactly 61,706 floats")  # Protect loaders.
    args.output.mkdir(parents=True, exist_ok=True)  # Create only the requested fixture destination.
    weight_path = args.output / "lenet_weights.bin"  # Use a plain nonexecutable model representation.
    weight_path.write_bytes(weights.astype("<f4").tobytes())  # Store portable little-endian float32.
    reference_path = args.output / "lenet_reference.bin"  # Keep framework logits as an independent oracle.
    reference_path.write_bytes(logits.numpy().astype("<f4").tobytes())  # Save all 10,000 test predictions.
    report = {  # Save enough information to reproduce and audit the preparation run.
        "architecture": "modern LeNet with full C3, ReLU, fixed average pooling and linear logits",  # Name changes.
        "parameters": int(weights.size),  # Record the exact native buffer length.
        "training_split": 60000,  # State how many training samples informed weights.
        "test_split": 10000,  # State the independent evaluation population.
        "seed": args.seed,  # Record initialization and permutation seed.
        "epochs": args.epochs,  # Record complete training passes.
        "batch_size": args.batch_size,  # Record optimizer batch size.
        "threads": args.threads,  # Record preparation CPU parallelism.
        "optimizer": "Adam",  # Describe the reference preparation update rule.
        "learning_rate": 0.001,  # Record its initial learning rate.
        "input_scale": "byte / 255 with a two-pixel zero border",  # Describe exact preprocessing.
        "training_cross_entropy": history,  # Keep per-epoch learning progress.
        "test_accuracy": accuracy,  # Keep the exact measured held-out accuracy.
        "test_cross_entropy": test_loss,  # Keep a second classification-quality measure.
        "training_seconds": training_seconds,  # Keep optional preparation timing.
        "torch_version": torch.__version__,  # Identify the optional training implementation.
        "numpy_version": np.__version__,  # Identify the array export implementation.
        "source_sha256": {path.name: digest(path) for path in paths},  # Identify all dataset inputs.
        "weights_sha256": digest(weight_path),  # Identify the actual deployed model bytes.
        "reference_sha256": digest(reference_path),  # Identify the independent test predictions.
    }  # Record the optional model-preparation environment.
    artifacts = [dict(entry) for entry in data_sources.PREPARED_ARTIFACTS]  # Preserve the prepared input records while updating the model export.
    model_paths = {weight_path.name: weight_path, reference_path.name: reference_path}  # Locate the two newly generated binary model artifacts.
    for entry in artifacts:  # Update model fingerprints and retain source attribution and byte layouts.
        if entry["name"] in model_paths:  # Leave every image, waveform, event and label fixture unchanged.
            path = model_paths[entry["name"]]  # Read only the newly exported model or framework reference.
            entry["bytes"] = path.stat().st_size  # Record the exact current binary export size.
            entry["sha256"] = digest(path)  # Bind the current model metadata to its actual exported bytes.
    data_sources.write_metadata(artifacts, report, args.metadata)  # Save current training provenance in the Python metadata module.
    print(f"held-out MNIST accuracy: {accuracy:.4%}")  # Report the complete test-split result.


if __name__ == "__main__":  # Run preparation only when this script is invoked explicitly.
    main()  # Train and export the requested portable example model.
