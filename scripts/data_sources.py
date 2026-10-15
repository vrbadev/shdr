"""Pinned public inputs and reproducible example-data metadata.

SOURCE_RECORDS contains download URLs, source fingerprints and attribution.
PREPARED_ARTIFACTS describes the ten data files used by the examples.
TRAINING_PROVENANCE describes the current independently trained LeNet export.
The archived training record retains the original metadata and fingerprint.
Verification uses the Python standard library. Preparation and training update
the current artifact and training assignments and retain the pinned sources.
"""

import ast  # Locate only the two metadata assignments that preparation may update.
import hashlib  # Check the archived training record against its original fingerprint.
import json  # Reconstruct the archived serialization in memory.
from pathlib import Path  # Resolve the maintained metadata module for updates.
import pprint  # Write current Python data with readable, deterministic formatting.

SCHEMA_VERSION = 1 # Version the record structure independently from prepared bytes.

SOURCE_RECORDS = [{'name': 't10k-images-idx3-ubyte.gz', # Keep all eight pinned downloads and their original attribution.
  'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz', # Download only this pinned public source.
  'bytes': 1648877, # Require this exact file size before accepting its contents.
  'sha256': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6', # Identify every byte of this recorded artifact.
  'dataset': 'MNIST', # Identify the public collection that supplied these values.
  'source_page': 'https://github.com/cvdfoundation/mnist', # Retain the original publisher attribution and context.
  'license': 'Not specified by the CVDF mirror', # Preserve the publisher's stated source notice.
  'license_source': 'http://yann.lecun.com/exdb/mnist/', # Link to the recorded terms or original author page.
  'credit': 'Yann LeCun, Corinna Cortes and Christopher J. C. Burges', # Retain attribution with redistributed sample data.
  'additional_license_notice': {'url': 'https://huggingface.co/datasets/ylecun/mnist/raw/main/README.md', # Keep this separately hosted copy notice distinct from the download mirror.
                                'declared_license': 'MIT', # Record only the notice on that separately hosted copy.
                                'scope': 'Dataset-card metadata for that hosted copy. The CVDF ' # Explain which published copy carries this notice.
                                         'mirror does not provide this notice.'}}, # Continue this recorded collection with its original values.
 {'name': 't10k-labels-idx1-ubyte.gz', # Continue this recorded collection with its original values.
  'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz', # Download only this pinned public source.
  'bytes': 4542, # Require this exact file size before accepting its contents.
  'sha256': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6', # Identify every byte of this recorded artifact.
  'dataset': 'MNIST', # Identify the public collection that supplied these values.
  'source_page': 'https://github.com/cvdfoundation/mnist', # Retain the original publisher attribution and context.
  'license': 'Not specified by the CVDF mirror', # Preserve the publisher's stated source notice.
  'license_source': 'http://yann.lecun.com/exdb/mnist/', # Link to the recorded terms or original author page.
  'credit': 'Yann LeCun, Corinna Cortes and Christopher J. C. Burges', # Retain attribution with redistributed sample data.
  'additional_license_notice': {'url': 'https://huggingface.co/datasets/ylecun/mnist/raw/main/README.md', # Keep this separately hosted copy notice distinct from the download mirror.
                                'declared_license': 'MIT', # Record only the notice on that separately hosted copy.
                                'scope': 'Dataset-card metadata for that hosted copy. The CVDF ' # Explain which published copy carries this notice.
                                         'mirror does not provide this notice.'}}, # Continue this recorded collection with its original values.
 {'name': 'train-images-idx3-ubyte.gz', # Continue this recorded collection with its original values.
  'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz', # Download only this pinned public source.
  'bytes': 9912422, # Require this exact file size before accepting its contents.
  'sha256': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609', # Identify every byte of this recorded artifact.
  'dataset': 'MNIST', # Identify the public collection that supplied these values.
  'source_page': 'https://github.com/cvdfoundation/mnist', # Retain the original publisher attribution and context.
  'license': 'Not specified by the CVDF mirror', # Preserve the publisher's stated source notice.
  'license_source': 'http://yann.lecun.com/exdb/mnist/', # Link to the recorded terms or original author page.
  'credit': 'Yann LeCun, Corinna Cortes and Christopher J. C. Burges', # Retain attribution with redistributed sample data.
  'additional_license_notice': {'url': 'https://huggingface.co/datasets/ylecun/mnist/raw/main/README.md', # Keep this separately hosted copy notice distinct from the download mirror.
                                'declared_license': 'MIT', # Record only the notice on that separately hosted copy.
                                'scope': 'Dataset-card metadata for that hosted copy. The CVDF ' # Explain which published copy carries this notice.
                                         'mirror does not provide this notice.'}}, # Continue this recorded collection with its original values.
 {'name': 'train-labels-idx1-ubyte.gz', # Continue this recorded collection with its original values.
  'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz', # Download only this pinned public source.
  'bytes': 28881, # Require this exact file size before accepting its contents.
  'sha256': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c', # Identify every byte of this recorded artifact.
  'dataset': 'MNIST', # Identify the public collection that supplied these values.
  'source_page': 'https://github.com/cvdfoundation/mnist', # Retain the original publisher attribution and context.
  'license': 'Not specified by the CVDF mirror', # Preserve the publisher's stated source notice.
  'license_source': 'http://yann.lecun.com/exdb/mnist/', # Link to the recorded terms or original author page.
  'credit': 'Yann LeCun, Corinna Cortes and Christopher J. C. Burges', # Retain attribution with redistributed sample data.
  'additional_license_notice': {'url': 'https://huggingface.co/datasets/ylecun/mnist/raw/main/README.md', # Keep this separately hosted copy notice distinct from the download mirror.
                                'declared_license': 'MIT', # Record only the notice on that separately hosted copy.
                                'scope': 'Dataset-card metadata for that hosted copy. The CVDF ' # Explain which published copy carries this notice.
                                         'mirror does not provide this notice.'}}, # Continue this recorded collection with its original values.
 {'name': 'nasa-blue-marble.jpg', # Continue this recorded collection with its original values.
  'url': 'https://assets.science.nasa.gov/dynamicimage/assets/science/psd/solar/2023/09/1/1-bluemarble_west-1.jpg?crop=faces%2Cfocalpoint&fit=clip&h=800&w=800', # Download only this pinned public source.
  'bytes': 105654, # Require this exact file size before accepting its contents.
  'sha256': '9824d30eca98594e73fd276388108ec1ee1ab494038e730fd378efc1aae4ffa6', # Identify every byte of this recorded artifact.
  'dataset': 'Blue Marble 2002', # Identify the public collection that supplied these values.
  'source_page': 'https://science.nasa.gov/resource/blue-marble-2002/', # Retain the original publisher attribution and context.
  'license': 'NASA media usage guidelines', # Preserve the publisher's stated source notice.
  'license_source': 'https://www.nasa.gov/nasa-brand-center/images-and-media/', # Link to the recorded terms or original author page.
  'credit': "NASA's Earth Observatory"}, # Retain attribution with redistributed sample data.
 {'name': 'mitdb-100.hea', # Continue this recorded collection with its original values.
  'url': 'https://physionet.org/files/mitdb/1.0.0/100.hea', # Download only this pinned public source.
  'bytes': 143, # Require this exact file size before accepting its contents.
  'sha256': '60ebc904c7bf3e04d142638d3fd5c903e8dc9f10f1ea3264e07926aa089ee75e', # Identify every byte of this recorded artifact.
  'dataset': 'MIT-BIH Arrhythmia Database record 100', # Identify the public collection that supplied these values.
  'source_page': 'https://physionet.org/content/mitdb/1.0.0/', # Retain the original publisher attribution and context.
  'license': 'ODC-By-1.0', # Preserve the publisher's stated source notice.
  'license_source': 'https://opendatacommons.org/licenses/by/1-0/', # Link to the recorded terms or original author page.
  'credit': 'George Moody and Roger Mark, MIT-BIH Arrhythmia Database version 1.0.0, PhysioNet, ' # Retain attribution with redistributed sample data.
            'DOI 10.13026/C2F305'}, # Continue this recorded collection with its original values.
 {'name': 'mitdb-100-prefix.dat', # Continue this recorded collection with its original values.
  'url': 'https://physionet.org/files/mitdb/1.0.0/100.dat', # Download only this pinned public source.
  'bytes': 6144, # Require this exact file size before accepting its contents.
  'range': 'bytes=0-6143', # Request only this bounded original byte interval.
  'source_bytes': 1950000, # Record the complete upstream file size alongside the selected byte range.
  'sha256': 'bfd10bda039ed650bf1a29a57cbd08df3994736c45c9ba1bc16e62fe97ca271a', # Identify every byte of this recorded artifact.
  'dataset': 'MIT-BIH Arrhythmia Database record 100', # Identify the public collection that supplied these values.
  'source_page': 'https://physionet.org/content/mitdb/1.0.0/', # Retain the original publisher attribution and context.
  'license': 'ODC-By-1.0', # Preserve the publisher's stated source notice.
  'license_source': 'https://opendatacommons.org/licenses/by/1-0/', # Link to the recorded terms or original author page.
  'credit': 'George Moody and Roger Mark, MIT-BIH Arrhythmia Database version 1.0.0, PhysioNet, ' # Retain attribution with redistributed sample data.
            'DOI 10.13026/C2F305'}, # Continue this recorded collection with its original values.
 {'name': 'laser-prefix.raw', # Continue this recorded collection with its original values.
  'url': 'https://kdrive.infomaniak.com/2/app/975517/share/71d66a09-e3b6-480b-ba94-a1509e8ab2c8/files/12198/download', # Download only this pinned public source.
  'bytes': 69632, # Require this exact file size before accepting its contents.
  'range': 'bytes=0-69631', # Request only this bounded original byte interval.
  'source_bytes': 341306299, # Record the complete upstream file size alongside the selected byte range.
  'sha256': 'a398733173c1e684b68ea737a448aebb0f59dec6aa9a76ba587bfca824b4d6aa', # Identify every byte of this recorded artifact.
  'dataset': 'Prophesee laser EVT3 sample', # Identify the public collection that supplied these values.
  'source_page': 'https://docs.prophesee.ai/stable/datasets.html', # Retain the original publisher attribution and context.
  'license': 'CC0-1.0', # Preserve the publisher's stated source notice.
  'license_source': 'https://creativecommons.org/publicdomain/zero/1.0/', # Link to the recorded terms or original author page.
  'credit': 'Prophesee sample recordings, laser, Gen4.1 sensor'}] # Retain attribution with redistributed sample data.

PREPARED_ARTIFACTS = [{'name': 'earth_rgb.bin', # Keep the ten physical runtime input and model fingerprints.
  'bytes': 12288, # Require this exact file size before accepting its contents.
  'sha256': 'ed28281973e480d946477c1f9bd83b0749c81cdd3ec7a15c7c422250b3f4cf5d', # Identify every byte of this recorded artifact.
  'format': 'RGB uint8, row-major, 64 by 64, three interleaved channels', # Describe the precise runtime byte layout.
  'source': 'nasa-blue-marble.jpg', # Trace the prepared file to its checked input or preparation script.
  'transform': 'JPEG decode to RGB, center-nearest downsample from 800 by 800 to 64 by 64', # Describe every conversion between the source and prepared values.
  'pillow_version': '10.2.0'}, # Identify the JPEG decoder used for these exact RGB bytes.
 {'name': 'signal_ecg.bin', # Continue this recorded collection with its original values.
  'bytes': 16384, # Require this exact file size before accepting its contents.
  'sha256': '9631961944e9e04e17722f44d522a216891b4c5763b348eb69d0160abfe3da44', # Identify every byte of this recorded artifact.
  'format': 'float32 little-endian, two channel-major arrays of 2048 samples', # Describe the precise runtime byte layout.
  'source': 'mitdb-100-prefix.dat', # Trace the prepared file to its checked input or preparation script.
  'transform': 'WFDB 212 decode, first 2048 sample pairs, (ADC - 1024) / 200 millivolts', # Describe every conversion between the source and prepared values.
  'sample_rate_hz': 360, # Interpret waveform positions at this original sampling rate.
  'channels': ['MLII', 'V5']}, # Preserve the lead names in channel-major storage order.
 {'name': 'evt3_words.bin', # Continue this recorded collection with its original values.
  'bytes': 65536, # Require this exact file size before accepting its contents.
  'sha256': 'da20cc642f8a3ba95bfcc3fe758cb7feed98334ffcfdf598c63fa2266775e014', # Identify every byte of this recorded artifact.
  'format': 'uint16 little-endian, 32768 EVT3 words', # Describe the precise runtime byte layout.
  'source': 'laser-prefix.raw', # Trace the prepared file to its checked input or preparation script.
  'source_byte_offset': 166, # Locate the first encoded event word after the original RAW header.
  'transform': 'Remove ASCII RAW header, retain first 32768 complete encoded words'}, # Describe every conversion between the source and prepared values.
 {'name': 'evt3_header.txt', # Continue this recorded collection with its original values.
  'bytes': 166, # Require this exact file size before accepting its contents.
  'sha256': 'a767d8bb9bce0446440d582b635cf8ba126390edd043d9f56f59f78ff63d865f', # Identify every byte of this recorded artifact.
  'format': 'ASCII RAW header', # Describe the precise runtime byte layout.
  'source': 'laser-prefix.raw', # Trace the prepared file to its checked input or preparation script.
  'transform': 'Copy the exact header preceding binary words'}, # Describe every conversion between the source and prepared values.
 {'name': 'mnist_train.bin', # Continue this recorded collection with its original values.
  'bytes': 100352, # Require this exact file size before accepting its contents.
  'sha256': 'b4e671ad6e9b4d02b6ef48194346fc857be35d2d0b9711d45913464271c1522e', # Identify every byte of this recorded artifact.
  'format': 'uint8, 128 row-major 28 by 28 images', # Describe the precise runtime byte layout.
  'source': 'train-images-idx3-ubyte.gz', # Trace the prepared file to its checked input or preparation script.
  'transform': 'First 64 instances each of digit 0 and 1 in original training order', # Describe every conversion between the source and prepared values.
  'source_indices': [1, # Preserve the exact zero-based MNIST training selections.
                     3, # Retain this original zero-based training-image index.
                     6, # Retain this original zero-based training-image index.
                     8, # Retain this original zero-based training-image index.
                     14, # Retain this original zero-based training-image index.
                     21, # Retain this original zero-based training-image index.
                     23, # Retain this original zero-based training-image index.
                     24, # Retain this original zero-based training-image index.
                     34, # Retain this original zero-based training-image index.
                     37, # Retain this original zero-based training-image index.
                     40, # Retain this original zero-based training-image index.
                     51, # Retain this original zero-based training-image index.
                     56, # Retain this original zero-based training-image index.
                     59, # Retain this original zero-based training-image index.
                     63, # Retain this original zero-based training-image index.
                     67, # Retain this original zero-based training-image index.
                     68, # Retain this original zero-based training-image index.
                     69, # Retain this original zero-based training-image index.
                     70, # Retain this original zero-based training-image index.
                     72, # Retain this original zero-based training-image index.
                     75, # Retain this original zero-based training-image index.
                     77, # Retain this original zero-based training-image index.
                     78, # Retain this original zero-based training-image index.
                     81, # Retain this original zero-based training-image index.
                     88, # Retain this original zero-based training-image index.
                     95, # Retain this original zero-based training-image index.
                     99, # Retain this original zero-based training-image index.
                     102, # Retain this original zero-based training-image index.
                     104, # Retain this original zero-based training-image index.
                     105, # Retain this original zero-based training-image index.
                     108, # Retain this original zero-based training-image index.
                     112, # Retain this original zero-based training-image index.
                     113, # Retain this original zero-based training-image index.
                     114, # Retain this original zero-based training-image index.
                     118, # Retain this original zero-based training-image index.
                     119, # Retain this original zero-based training-image index.
                     121, # Retain this original zero-based training-image index.
                     124, # Retain this original zero-based training-image index.
                     128, # Retain this original zero-based training-image index.
                     134, # Retain this original zero-based training-image index.
                     152, # Retain this original zero-based training-image index.
                     156, # Retain this original zero-based training-image index.
                     169, # Retain this original zero-based training-image index.
                     174, # Retain this original zero-based training-image index.
                     177, # Retain this original zero-based training-image index.
                     184, # Retain this original zero-based training-image index.
                     192, # Retain this original zero-based training-image index.
                     200, # Retain this original zero-based training-image index.
                     201, # Retain this original zero-based training-image index.
                     205, # Retain this original zero-based training-image index.
                     206, # Retain this original zero-based training-image index.
                     208, # Retain this original zero-based training-image index.
                     209, # Retain this original zero-based training-image index.
                     210, # Retain this original zero-based training-image index.
                     211, # Retain this original zero-based training-image index.
                     216, # Retain this original zero-based training-image index.
                     224, # Retain this original zero-based training-image index.
                     229, # Retain this original zero-based training-image index.
                     231, # Retain this original zero-based training-image index.
                     232, # Retain this original zero-based training-image index.
                     234, # Retain this original zero-based training-image index.
                     246, # Retain this original zero-based training-image index.
                     248, # Retain this original zero-based training-image index.
                     249, # Retain this original zero-based training-image index.
                     251, # Retain this original zero-based training-image index.
                     260, # Retain this original zero-based training-image index.
                     269, # Retain this original zero-based training-image index.
                     270, # Retain this original zero-based training-image index.
                     276, # Retain this original zero-based training-image index.
                     283, # Retain this original zero-based training-image index.
                     290, # Retain this original zero-based training-image index.
                     293, # Retain this original zero-based training-image index.
                     296, # Retain this original zero-based training-image index.
                     303, # Retain this original zero-based training-image index.
                     309, # Retain this original zero-based training-image index.
                     310, # Retain this original zero-based training-image index.
                     315, # Retain this original zero-based training-image index.
                     320, # Retain this original zero-based training-image index.
                     326, # Retain this original zero-based training-image index.
                     345, # Retain this original zero-based training-image index.
                     351, # Retain this original zero-based training-image index.
                     355, # Retain this original zero-based training-image index.
                     357, # Retain this original zero-based training-image index.
                     358, # Retain this original zero-based training-image index.
                     359, # Retain this original zero-based training-image index.
                     366, # Retain this original zero-based training-image index.
                     382, # Retain this original zero-based training-image index.
                     394, # Retain this original zero-based training-image index.
                     397, # Retain this original zero-based training-image index.
                     398, # Retain this original zero-based training-image index.
                     399, # Retain this original zero-based training-image index.
                     406, # Retain this original zero-based training-image index.
                     408, # Retain this original zero-based training-image index.
                     416, # Retain this original zero-based training-image index.
                     427, # Retain this original zero-based training-image index.
                     429, # Retain this original zero-based training-image index.
                     435, # Retain this original zero-based training-image index.
                     440, # Retain this original zero-based training-image index.
                     443, # Retain this original zero-based training-image index.
                     447, # Retain this original zero-based training-image index.
                     450, # Retain this original zero-based training-image index.
                     451, # Retain this original zero-based training-image index.
                     453, # Retain this original zero-based training-image index.
                     454, # Retain this original zero-based training-image index.
                     455, # Retain this original zero-based training-image index.
                     458, # Retain this original zero-based training-image index.
                     462, # Retain this original zero-based training-image index.
                     464, # Retain this original zero-based training-image index.
                     466, # Retain this original zero-based training-image index.
                     470, # Retain this original zero-based training-image index.
                     473, # Retain this original zero-based training-image index.
                     475, # Retain this original zero-based training-image index.
                     484, # Retain this original zero-based training-image index.
                     489, # Retain this original zero-based training-image index.
                     519, # Retain this original zero-based training-image index.
                     524, # Retain this original zero-based training-image index.
                     526, # Retain this original zero-based training-image index.
                     527, # Retain this original zero-based training-image index.
                     542, # Retain this original zero-based training-image index.
                     577, # Retain this original zero-based training-image index.
                     582, # Retain this original zero-based training-image index.
                     596, # Retain this original zero-based training-image index.
                     603, # Retain this original zero-based training-image index.
                     612, # Retain this original zero-based training-image index.
                     633, # Retain this original zero-based training-image index.
                     639, # Retain this original zero-based training-image index.
                     656, # Retain this original zero-based training-image index.
                     662]}, # Continue this recorded collection with its original values.
 {'name': 'mnist_train_labels.bin', # Continue this recorded collection with its original values.
  'bytes': 128, # Require this exact file size before accepting its contents.
  'sha256': '49994fd8dc3bf88c7dff6d9863bde8bbf0d96ff09b8e99b3aaa4fbfceab5d3bf', # Identify every byte of this recorded artifact.
  'format': 'uint8, 128 digit labels', # Describe the precise runtime byte layout.
  'source': 'train-labels-idx1-ubyte.gz', # Trace the prepared file to its checked input or preparation script.
  'transform': 'Labels at the indices recorded in mnist_train.bin'}, # Describe every conversion between the source and prepared values.
 {'name': 'mnist_test.bin', # Continue this recorded collection with its original values.
  'bytes': 401408, # Require this exact file size before accepting its contents.
  'sha256': '80d1589bb8ec86cbe5b156dc437990d35e226cf0a5c7dc8202fc5e58124eff1d', # Identify every byte of this recorded artifact.
  'format': 'uint8, 512 row-major 28 by 28 images', # Describe the precise runtime byte layout.
  'source': 't10k-images-idx3-ubyte.gz', # Trace the prepared file to its checked input or preparation script.
  'transform': 'Remove IDX header, retain first 512 official test images'}, # Describe every conversion between the source and prepared values.
 {'name': 'mnist_labels.bin', # Continue this recorded collection with its original values.
  'bytes': 512, # Require this exact file size before accepting its contents.
  'sha256': '828c90776b296fc6bbadc357389e4d66dd7819fbb1979982a700d983168da942', # Identify every byte of this recorded artifact.
  'format': 'uint8, 512 digit labels', # Describe the precise runtime byte layout.
  'source': 't10k-labels-idx1-ubyte.gz', # Trace the prepared file to its checked input or preparation script.
  'transform': 'Remove IDX header, retain first 512 official test labels'}, # Describe every conversion between the source and prepared values.
 {'name': 'lenet_weights.bin', # Continue this recorded collection with its original values.
  'bytes': 246824, # Require this exact file size before accepting its contents.
  'sha256': '327dd259df961e6ed9b8f9fc82d2cd56b1334ca7525f92403de9c6495d1fe8de', # Identify every byte of this recorded artifact.
  'format': 'float32 little-endian, 61706 LeNet parameters', # Describe the precise runtime byte layout.
  'source': 'scripts/train_lenet.py', # Trace the prepared file to its checked input or preparation script.
  'transform': 'Offline full-MNIST training and reference export described in lenet_training.json'}, # Describe every conversion between the source and prepared values.
 {'name': 'lenet_reference.bin', # Continue this recorded collection with its original values.
  'bytes': 400000, # Require this exact file size before accepting its contents.
  'sha256': 'e3623237df15d5671dbb5cfd04c072d58f3d14a31bc04fd6b80c1d29026195e5', # Identify every byte of this recorded artifact.
  'format': 'float32 little-endian, 10000 by 10 official-test reference logits', # Describe the precise runtime byte layout.
  'source': 'scripts/train_lenet.py', # Trace the prepared file to its checked input or preparation script.
  'transform': 'Offline full-MNIST training and reference export described in lenet_training.json'}] # Describe every conversion between the source and prepared values.

ARCHIVED_TRAINING_ARTIFACT = {'name': 'lenet_training.json', # Retain the original provenance-file record after moving its contents here.
 'bytes': 1252, # Require this exact file size before accepting its contents.
 'sha256': '02798d534978d33d7805ba04e6143eab77849853c1bc30d602e7de9726561dda', # Identify every byte of this recorded artifact.
 'format': 'JSON training settings, dataset checksums and independent metrics', # Describe the precise runtime byte layout.
 'source': 'scripts/train_lenet.py', # Trace the prepared file to its checked input or preparation script.
 'transform': 'Offline full-MNIST training and reference export described in lenet_training.json'} # Describe every conversion between the source and prepared values.

ARCHIVED_TRAINING_PROVENANCE = {'architecture': 'modern LeNet with full C3, ReLU, fixed average pooling and linear logits', # Preserve every value from the original independent training export.
 'parameters': 61706, # Match the flat native parameter count.
 'training_split': 60000, # Record the number of original training examples.
 'test_split': 10000, # Keep the independent held-out evaluation population explicit.
 'seed': 12345, # Reproduce parameter initialization and training permutations.
 'epochs': 3, # Visit the complete training split this many times.
 'batch_size': 128, # Bound each Adam update to this many training images.
 'threads': 8, # Record the model-preparation thread count.
 'optimizer': 'Adam', # Identify the independent reference training update rule.
 'learning_rate': 0.001, # Record the Adam learning rate.
 'input_scale': 'byte / 255 with a two-pixel zero border', # Match the native image normalization and convolution border.
 'training_cross_entropy': [0.4789702771107356, 0.12624290428757667, 0.08441878777146339], # Preserve the measured mean loss after each complete training epoch.
 'test_accuracy': 0.9778000116348267, # Keep the exact measured held-out accuracy.
 'test_cross_entropy': 0.06259464472532272, # Preserve the independently measured held-out loss.
 'training_seconds': 18.704168565000145, # Record the completed original preparation time.
 'torch_version': '2.6.0+cpu', # Identify the optional reference training framework.
 'numpy_version': '1.26.4', # Identify the optional float-array exporter.
 'source_sha256': {'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609', # Identify all four original archives used during training and evaluation.
                   'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c', # Preserve this recorded source fingerprint.
                   't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6', # Preserve this recorded source fingerprint.
                   't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'}, # Preserve this recorded source fingerprint.
 'weights_sha256': '327dd259df961e6ed9b8f9fc82d2cd56b1334ca7525f92403de9c6495d1fe8de', # Identify the deployed flat model parameters.
 'reference_sha256': 'e3623237df15d5671dbb5cfd04c072d58f3d14a31bc04fd6b80c1d29026195e5'} # Identify all independent held-out framework logits.

TRAINING_PROVENANCE = ARCHIVED_TRAINING_PROVENANCE # Start with the archived model and replace this assignment when retraining.


def archived_training_bytes(): # Reconstruct the former standalone provenance file for its historical integrity check.
    """Return the original serialized provenance bytes."""
    return (json.dumps(ARCHIVED_TRAINING_PROVENANCE, indent=2) + "\n").encode() # Preserve the original UTF-8 bytes and final newline.


def verify_archived_training(): # Keep the migrated record as verifiable as the former standalone file.
    """Check the original training metadata against its retained size and SHA-256."""
    payload = archived_training_bytes() # Serialize the archived original training record.
    record = ARCHIVED_TRAINING_ARTIFACT # Read the exact historical fingerprint copied during migration.
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]: # Detect changed archived values or serialization.
        raise ValueError("archived LeNet training provenance fingerprint differs") # Keep accidental provenance edits visible during offline verification.


def _assignment(name, value): # Format the Python metadata assignment.
    """Render ordinary Python values with brief field comments on each code line."""
    lines = pprint.pformat(value, width=100, sort_dicts=False).splitlines() # Keep source and artifact ordering stable between updates.
    result = [] # Collect one commented source line at a time.
    for index, line in enumerate(lines): # Preserve the formatter's readable nested record layout.
        stripped = line.lstrip() # Inspect a key and preserve its emitted indentation.
        comment = "Continue this recorded collection." # Describe wrapped record values and closing delimiters.
        if stripped.startswith("'") and "':" in stripped: # A dictionary key names the value on this line.
            key = stripped.split("':", 1)[0][1:] # Read the Python string key through literal parsing.
            comment = "Record " + key.replace("_", " ") + "." # Make each emitted metadata field easy to identify.
        elif stripped.rstrip(",").isdigit(): # The selected training rows are stored as integer list entries.
            comment = "Retain this original zero-based training-image index." # Explain why the individual numbers are preserved.
        result.append((name + " = " if index == 0 else "") + line + " # " + comment) # Write a valid assignment with line comments.
    return "\n".join(result) + "\n" # End the assignment with one normal source newline.


def write_metadata(artifacts, training=None, path=None): # Update current outputs while retaining pinned sources and archived provenance.
    """Write current prepared-file records and optional training results.

    An alternate target starts from this module's source, allowing preparation
    checks in a separate metadata module.
    """
    target = Path(path) if path is not None else Path(__file__).resolve() # Default to the single maintained metadata module.
    source = target.read_text() if target.exists() else Path(__file__).resolve().read_text() # Copy the maintained module when checking another output directory.
    tree = ast.parse(source) # Find assignment boundaries through the Python syntax tree.
    replacements = {"PREPARED_ARTIFACTS": artifacts} # Always record the current ten prepared physical outputs.
    if training is not None: # A training run also supplies its new measured model provenance.
        replacements["TRAINING_PROVENANCE"] = training # Update current results and retain the archived original run.
    spans = [] # Apply edits from the end so earlier line positions remain valid.
    for statement in tree.body: # Examine only the module's top-level assignments.
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name): # Ignore helpers and complex Python statements.
            name = statement.targets[0].id # Read the constant being assigned on this source span.
            if name in replacements: # Restrict writes to the explicit current metadata fields.
                spans.append((statement.lineno - 1, statement.end_lineno, _assignment(name, replacements.pop(name)))) # Retain complete assignment lines including comments.
    if replacements: # Require every requested assignment in the metadata module.
        raise ValueError("metadata assignments are missing: " + ", ".join(replacements)) # Identify the required metadata assignment.
    lines = source.splitlines(keepends=True) # Preserve all source text outside the two selected assignments.
    for start, end, replacement in reversed(spans): # Replace the later assignment before the earlier one.
        lines[start:end] = [replacement] # Substitute only the current model or prepared-file record.
    updated = "".join(lines) # Reassemble the metadata with its original surrounding documentation.
    if updated != source or not target.exists(): # Preserve timestamps while still creating an explicitly selected new destination.
        target.parent.mkdir(parents=True, exist_ok=True) # Permit an explicitly selected new metadata destination.
        target.write_text(updated) # Store the updated records as readable importable Python.
