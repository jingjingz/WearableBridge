# WearableBridge

Code for training Mapper, MeanFlow, and Distillation bridges on features from
your own encoder and dataset.

## Files

| File | Contents |
| --- | --- |
| `wearablebridge.py` | Bridge models, losses, normalization, and prediction |
| `train.py` | Training and checkpoint loading |
| `requirements.txt` | Python dependencies |
| `README.md` | Instructions |
| `index.html` | Links to the interactive figures |
| `thigh2wrist.html` | Thigh2Wrist figure |
| `lab2freeliving.html` | Lab2Freeliving figure |

## Steps

### 1. Install

Use Python 3.12 in a virtual environment. Run these commands from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional CPU check with synthetic features:

```python
import tempfile
from pathlib import Path
import torch
from train import train_bridge, load_bridge
from wearablebridge import predict

torch.set_num_threads(1)
torch.manual_seed(0)
labels = torch.arange(32) % 4
reference = torch.eye(4)[labels] + 0.05 * torch.randn(32, 4)
source = reference + 0.2
classifier = torch.nn.Linear(4, 4)
with torch.no_grad():
    classifier.weight.copy_(torch.eye(4))
    classifier.bias.zero_()
batches = [(source, reference, labels, torch.ones(32))]
checkpoint = train_bridge(batches, reference, classifier,
                          method="meanflow", hidden_dim=16, epochs=2, seed=0)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "bridge.pt"
    torch.save(checkpoint, path)
    bridge, head, standardizer = load_bridge(path)
    probabilities = predict(source, bridge, head, standardizer)
assert probabilities.shape == (32, 4) and torch.isfinite(probabilities).all()
print("Training and prediction passed.")
```

### 2. Prepare features

Fit a classifier on reference-training data, then freeze it and the encoder.
Extract source and reference features before standardization, using the same
encoder and fitted projection. The classifier must return logits from those
features. Split participants into training, validation, and test sets before
pairing. Pair training examples using
simultaneous recordings or matches within participant, activity, and wrist-side
groups. Each source example needs an activity label.

Save a dictionary as `calibration.pt` using `torch.save`. For command-line
training, the classifier must be `torch.nn.Linear(D, C)`, where D is the feature
dimension and C is the number of classes.

| Key | Value |
| --- | --- |
| `source` | Float32 `[N, D]` source features |
| `reference` | Float32 `[N, D]` paired reference features |
| `labels` | Int64 `[N]` source labels, numbered `0` to `C-1` |
| `reference_train` | Float32 `[M, D]` unique reference-training features for normalization |
| `classifier` | Fitted classifier's `state_dict()` |
| `weights` | Optional positive float32 `[N]` pair weights, defaulting to ones |

### 3. Train

```bash
python train.py --data calibration.pt --output bridge.pt --method meanflow
```

Use `--method mapper` or `--method distillation` to change the method, and
`--device cuda` to use a GPU. Run `python train.py --help` for other options.

By default, training runs for 30 epochs and saves the final model. To select by
validation macro-F1 with early stopping, add `--validation validation.pt`.
That file contains `source` and `labels` for unique validation windows, with
the same format as above.

For a custom classifier or new pairs each epoch, use `train_bridge` in
`train.py`. It accepts batches of `(source, reference, labels, weights)` or a
function `batches(epoch)` that returns them, starting at epoch 1. Seed and
recreate your loader or pairing function for each run. Pair losses use
`mean(weight * loss)` with training-set group weights. Validation batches
contain `(source, labels)`.

```python
import torch
from train import train_bridge

checkpoint = train_bridge(batches, reference_train, classifier,
                          validation=validation_batches, seed=20260717)
torch.save(checkpoint, "bridge.pt")
```

### 4. Predict

```python
from train import load_bridge
from wearablebridge import predict

bridge, classifier, standardizer = load_bridge("bridge.pt")
probabilities = predict(source_features.cpu().float(), bridge, classifier, standardizer)
labels = probabilities.argmax(dim=1)
```

Use the same encoder, preprocessing, and class order as in training. For a
custom classifier, pass its architecture to `load_bridge(..., classifier=model)`.
Predict large datasets in batches. For the paper's five-window results:

```python
from wearablebridge import causal_average

probabilities = causal_average(probabilities, stream, start_us, window_count=5)
labels = probabilities.argmax(dim=1)
```

`stream` is an integer ID for each participant/wrist recording stream and
`start_us` contains integer window-start times in microseconds. Include all
windows before applying the activity scoring mask. History resets at gaps.

### 5. View the figures

Download this repository and open `index.html` in a browser. Both figures work offline. Playback interpolates
between projected endpoints. Intermediate frames are illustrative.

## Encoders used in the paper

The pretrained encoder sources and checkpoint names are linked below.

| Encoder | Paper | Weights |
| --- | --- | --- |
| MoCA | Ryu et al. (2025), arXiv:2506.02260 | [MoCA code](https://github.com/HowonRyu/MoCA), [checkpoint-19.pth](https://github.com/HowonRyu/MoCA/blob/main/checkpoints/checkpoint-19.pth) |
| HARNet10 | Yuan et al. (2024), npj Digital Medicine, doi:10.1038/s41746-024-01062-3 | [HARNet code and loading instructions](https://github.com/OxWearables/ssl-wearables), `mtl_best.mdl` |
| DeepConvLSTM | Ordóñez and Roggen (2016), Sensors, doi:10.3390/s16010115 | Local implementation, trained from scratch |
| InceptionTime | Ismail Fawaz et al. (2020), Data Mining and Knowledge Discovery, doi:10.1007/s10618-020-00710-y | Local implementation, trained from scratch |
| HART | Ek et al. (2023), Personal and Ubiquitous Computing, doi:10.1007/s00779-023-01776-3 | Local accelerometer-only adaptation, trained from scratch |

Reference classifiers and the MoCA projection are fitted on reference data.
They are not included in the downloaded weights. Encoder code, datasets, and
checkpoints are not bundled here.

For primary four-class MoCA Lab2Freeliving, use 128-dimensional features,
`--hidden-dim 256`, `--epsilon 0.01`, and validation selection. The features
include the fitted linear projection, LayerNorm, and GELU, followed by a
frozen `Linear(128, 4)` classifier. Rematch pairs each epoch using the Python
interface, with the same schedule across methods. Within each participant,
activity, and wrist-side group, independently shuffle both lists and cycle
the shorter list, then shuffle all pairs together.

Thigh2Wrist uses 768-dimensional MoCA features, a frozen `Linear(768, 4)`
classifier, `--hidden-dim 768`, `--epsilon 0.06`, and the final epoch.
Both designs use batches of 128, learning rate and weight decay `1e-4`, and
30 epochs maximum. Paper seeds are `20260717`, `20260718`, and `20260719`.
Pass `--seed` to change the CLI default of 0. See the manuscript appendix for
preprocessing, reference fitting, participant splits, and evaluation.

## Paper

Jingjing Zou and Yixin Wang (2026). *WearableBridge: Task-Guided Latent
Transport for Free-Living Activity Recognition*.

## License

The bridge code and demo viewer are released under the [MIT License](LICENSE).
External encoders and datasets retain their respective licenses and access terms.
