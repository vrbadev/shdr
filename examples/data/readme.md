# Public data used by the examples

These ten prepared files supply recorded images, signals, events and handwritten
digits for `example_real_data`. The executable reads their binary and text
formats directly for CPU and GPU processing.

[The metadata module](../../scripts/data_sources.py) holds the eight pinned
upstream downloads, ten prepared-file fingerprints and complete model-training
provenance. Use it to look up exact source selections, file layouts and training
settings when adapting the examples to your own data.

From the project root, verify the supplied files, then run the examples:

```sh
python3 scripts/prepare_datasets.py --verify
./build/example_real_data
./build/example_real_data --gpu es31
```

Verification checks local file sizes and hashes with Python's standard library.
The ten runtime files have the layouts and fingerprints described below.

## Satellite image

`earth_rgb.bin` contains 64 by 64 RGB pixels as unsigned bytes, row by row, with
three interleaved channels. It is 12,288 bytes. The source is the 800 by 800 JPEG
served by NASA's [Blue Marble 2002 page](https://science.nasa.gov/resource/blue-marble-2002/):

[Original JPEG download](https://assets.science.nasa.gov/dynamicimage/assets/science/psd/solar/2023/09/1/1-bluemarble_west-1.jpg?crop=faces%2Cfocalpoint&fit=clip&h=800&w=800).

The checked download is 105,654 bytes with SHA-256
`9824d30eca98594e73fd276388108ec1ee1ab494038e730fd378efc1aae4ffa6`.
Credit: **NASA's Earth Observatory**. The applicable source notice is the
[NASA images and media guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/).

Preparation decodes the JPEG to RGB with Pillow 10.2.0. Each output pixel takes
source coordinate `floor((2*i + 1) * 800 / 128)` on each axis. Sampling retains
the decoded RGB channel values. The example normalizes bytes when it needs float
channels. This image feeds buffer chains, blur, corners, convolution, Fourier
transforms, asynchronous jobs and compression checks.

Bayer inputs are mosaics sampled from these RGB pixels. Corner checks compare
numerical references and CPU/GPU outputs. RLE checks exact byte reconstruction
and reports the input and encoded sizes.

## Two-channel ECG recording

`signal_ecg.bin` contains two channel-major arrays of 2,048 little-endian float32
samples. The first array is lead MLII, and the second is V5. Values are in
millivolts at the original 360 Hz sample rate. The file is 16,384 bytes.

The source is George Moody and Roger Mark's
[MIT-BIH Arrhythmia Database, version 1.0.0](https://physionet.org/content/mitdb/1.0.0/),
record 100, DOI **10.13026/C2F305**, distributed through PhysioNet under the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
Retain that attribution when redistributing the prepared excerpt.

| Original input | Selected bytes | SHA-256 |
| --- | ---: | --- |
| [100.hea](https://physionet.org/files/mitdb/1.0.0/100.hea) | Complete 143-byte header | `60ebc904c7bf3e04d142638d3fd5c903e8dc9f10f1ea3264e07926aa089ee75e` |
| [100.dat](https://physionet.org/files/mitdb/1.0.0/100.dat) | HTTP range `bytes=0-6143` from the 1,950,000-byte recording | `bfd10bda039ed650bf1a29a57cbd08df3994736c45c9ba1bc16e62fe97ca271a` |

Preparation decodes the first 2,048 pairs from WFDB format 212, sign-extends each
12-bit sample, and applies `(ADC - 1024) / 200` using the original header's
calibration. It stores all MLII samples before all V5 samples. The FFT, inverse
FFT, frequency masks, windows, FIR and stateful biquad cases process these values.
The checks measure waveform reconstruction and filter-output agreement on the
recorded samples.

## Event-camera recording

`evt3_words.bin` contains the first 32,768 encoded words after the original RAW
header. Words are little-endian uint16 and occupy 65,536 bytes.
`evt3_header.txt` contains the exact preceding 166-byte ASCII header.

The source is the **Prophesee laser sample, Gen4.1 sensor**, listed in the official
[Prophesee sample recordings](https://docs.prophesee.ai/stable/datasets.html).
That collection identifies its sample data as
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

[Original RAW download](https://kdrive.infomaniak.com/2/app/975517/share/71d66a09-e3b6-480b-ba94-a1509e8ab2c8/files/12198/download).
The complete source is 341,306,299 bytes. Preparation requests HTTP range
`bytes=0-69631`, a 69,632-byte prefix with SHA-256
`a398733173c1e684b68ea737a448aebb0f59dec6aa9a76ba587bfca824b4d6aa`.
It preserves the original header, encoded words, timestamps and coordinates.

The fixture decodes to 21,950 CD records. Event positions are sensor pixel
coordinates, and decoded timestamps use microseconds. With a three-pixel radius,
128-event lookback and 10,000-microsecond time window, 500 records have suitable
neighbors for accepted normal-flow fits. These numbers measure decoder output
and fit acceptance. Flow validation compares independent temporal-plane fits and
CPU/GPU results.

## MNIST digit images and labels

MNIST is credited to **Yann LeCun, Corinna Cortes and Christopher J. C. Burges**.
The [CVDF mirror](https://github.com/cvdfoundation/mnist) records permission to
host the original files. The [original author page](http://yann.lecun.com/exdb/mnist/)
provides source information and credits. A separately hosted
[MNIST dataset card](https://huggingface.co/datasets/ylecun/mnist/raw/main/README.md)
declares MIT for that copy. The metadata identifies the hosting-permission notice
and the dataset-card declaration separately for their respective copies.

| Original archive | Bytes | SHA-256 |
| --- | ---: | --- |
| [train-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz) | 9,912,422 | `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609` |
| [train-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz) | 28,881 | `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c` |
| [t10k-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz) | 1,648,877 | `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6` |
| [t10k-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz) | 4,542 | `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6` |

The archives contain 60,000 training images and 10,000 independent test images.
Preparation validates the IDX headers and removes them. Pixels remain unsigned
bytes in row-major 28 by 28 order. Labels are digit class bytes from zero to nine.

| Prepared file | Selection and layout |
| --- | --- |
| `mnist_train.bin` | 128 images, 100,352 bytes. First 64 instances each of digits zero and one, retained in original training order |
| `mnist_train_labels.bin` | 128 matching labels, 128 bytes |
| `mnist_test.bin` | First 512 official test images, 401,408 bytes |
| `mnist_labels.bin` | First 512 official test labels, 512 bytes |

The metadata records all 128 original training indices. Native neural inputs use
`pixel / 255`. The small ANN derives two stroke features, the small CNN samples
an 8 by 8 image, and the transformer forms four tokens from sixteen 7 by 7 patch
means. Their small training examples fit selected training batches. The 3D
convolution input is a stack of digit images along a depth axis.

## Prepared LeNet model and independent scores

`lenet_weights.bin` holds **61,706 little-endian float32 parameters** in 246,824
bytes. `lenet_reference.bin` holds **10,000 by 10 little-endian float32 logits**
in 400,000 bytes, in original official test order. Both are generated by
[train_lenet.py](../../scripts/train_lenet.py), with full settings and results in
`TRAINING_PROVENANCE` inside [data_sources.py](../../scripts/data_sources.py).

The model is a modern LeNet variant with full second-layer channel connectivity,
ReLU activations, fixed average pooling and linear output logits. It supplies a
two-pixel zero border around each 28 by 28 input, then runs seven native stages:

| Stage | Output per image |
| --- | --- |
| 5 by 5 convolution, 1 to 6 channels, ReLU | 6 by 28 by 28 |
| 2 by 2 average pool, stride 2 | 6 by 14 by 14 |
| 5 by 5 convolution, 6 to 16 channels, ReLU | 16 by 10 by 10 |
| 2 by 2 average pool, stride 2 | 16 by 5 by 5 |
| Dense 400 to 120, ReLU | 120 values |
| Dense 120 to 84, ReLU | 84 values |
| Dense 84 to 10 | 10 raw class scores |

Convolution weights use `[out,in,ky,kx]` order and dense weights use `[out,in]`
order. Each layer's biases immediately follow its weights. Weight and bias
offsets are respectively 0/150, 156/2556, 2572/50572, 50692/60772 and 60856/61696.
The predicted class is the index of the greatest logit.

The recorded preparation used all 60,000 training images for three epochs, Adam
with learning rate 0.001, batches of 128, seed 12345 and eight CPU threads.
Preparation versions were PyTorch 2.6.0+cpu and NumPy 1.26.4. Training mean cross
entropies were 0.4789702771107356, 0.12624290428757667 and 0.08441878777146339.
Independent evaluation used all 10,000 held-out images and measured 97.78%
accuracy, with cross entropy 0.06259464472532272. Native examples compare logits
against the independent scores and test accuracy against the original labels.
Model preparation uses optional Python packages. Deployed inference uses the
C++/GLSL pipeline and plain float arrays.

## Prepared-file fingerprints

| File | SHA-256 |
| --- | --- |
| `earth_rgb.bin` | `ed28281973e480d946477c1f9bd83b0749c81cdd3ec7a15c7c422250b3f4cf5d` |
| `signal_ecg.bin` | `9631961944e9e04e17722f44d522a216891b4c5763b348eb69d0160abfe3da44` |
| `evt3_words.bin` | `da20cc642f8a3ba95bfcc3fe758cb7feed98334ffcfdf598c63fa2266775e014` |
| `evt3_header.txt` | `a767d8bb9bce0446440d582b635cf8ba126390edd043d9f56f59f78ff63d865f` |
| `mnist_train.bin` | `b4e671ad6e9b4d02b6ef48194346fc857be35d2d0b9711d45913464271c1522e` |
| `mnist_train_labels.bin` | `49994fd8dc3bf88c7dff6d9863bde8bbf0d96ff09b8e99b3aaa4fbfceab5d3bf` |
| `mnist_test.bin` | `80d1589bb8ec86cbe5b156dc437990d35e226cf0a5c7dc8202fc5e58124eff1d` |
| `mnist_labels.bin` | `828c90776b296fc6bbadc357389e4d66dd7819fbb1979982a700d983168da942` |
| `lenet_weights.bin` | `327dd259df961e6ed9b8f9fc82d2cd56b1334ca7525f92403de9c6495d1fe8de` |
| `lenet_reference.bin` | `e3623237df15d5671dbb5cfd04c072d58f3d14a31bc04fd6b80c1d29026195e5` |

## Reproduce preparation and training

From the project root, optional preparation uses Pillow for the JPEG. Downloading
occurs only with `--fetch`. Cached sources are size- and SHA-256-checked before
conversion, and recording downloads require the server to honor the exact range.

```sh
# Download missing checked inputs and prepare the sample data.
python3 scripts/prepare_datasets.py --fetch
# Reuse the checked local source cache.
python3 scripts/prepare_datasets.py
# Also unpack every official test image and label under build/datasets.
python3 scripts/prepare_datasets.py --full-mnist
# Verify prepared files using only Python's standard library.
python3 scripts/prepare_datasets.py --verify
```

The default source cache is `build/datasets/upstream`. `--cache` chooses another
source cache, `--output` chooses the prepared-data directory, and `--full-output`
chooses the complete test-split directory. To prepare a separate data directory
with the supplied model and full test split:

```sh
mkdir -p build/my-data
cp examples/data/lenet_weights.bin examples/data/lenet_reference.bin build/my-data/
python3 scripts/prepare_datasets.py --cache build/datasets/upstream \
  --output build/my-data --full-output build/my-data/full \
  --metadata build/my-data/data_sources.py --full-mnist
```

Preparation creates the eight recorded-input files. Copy the supplied model
exports or train the model to add the two classifier files.

Retraining uses PyTorch and NumPy. The command checks the same four pinned MNIST
archives and writes the two
plain binary exports, together with training settings, metrics and fingerprints:

```sh
python3 scripts/train_lenet.py --cache build/datasets/upstream \
  --epochs 3 --batch-size 128 --seed 12345 --threads 8
python3 scripts/prepare_datasets.py --verify
```

`--output` and `--metadata` can keep a new model and its metadata outside this
directory. Framework versions and execution environment affect floating-point
training results. Use the exported fingerprints to identify each trained model.

After full preparation, evaluate all 10,000 held-out images with:

```sh
SHDR_FULL_MNIST_DIR=build/datasets ./build/example_real_data
SHDR_FULL_MNIST_DIR=build/datasets ./build/example_real_data --gpu es31
```

`SHDR_DATA_DIR` selects an alternate small-fixture and model directory. These
end-to-end evaluation timings include batch construction, reference comparisons
and result reads. The benchmark tables measure completed resident pipeline
operations.
