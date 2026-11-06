# WearableBridge

**Task-Guided Latent Transport for Free-Living Activity Recognition**

Jingjing Zou and Yixin Wang, 2026

WearableBridge uses reference recordings to improve activity recognition from
wrist accelerometers. It learns a small bridge that maps wrist features into
the feature space of a trained reference classifier. The encoder and reference
classifier stay fixed, and prediction uses wrist input alone.

The code implements **Mapper, MeanFlow, and Distillation**, with training and
inference from extracted features and a fitted reference classifier.

## Explore the figures

Open the [demo page](https://jingjingz.github.io/WearableBridge/) or go directly
to either figure:

- **[Thigh2Wrist](https://jingjingz.github.io/WearableBridge/thigh2wrist.html)**
  compares wrist features before and after bridging with features from concurrent
  thigh recordings.
- **[Lab2Freeliving](https://jingjingz.github.io/WearableBridge/lab2freeliving.html)**
  compares the three bridges with independently recorded laboratory wrist references.

The figures require no Python installation. For offline use, download the
repository and open `index.html` in a browser.

Playback interpolates between projected endpoints. Intermediate
frames illustrate the change and are not model trajectories.

## Install

Use Python 3.12 with the pinned dependencies in `requirements.txt`.

```bash
git clone https://github.com/jingjingz/WearableBridge.git
cd WearableBridge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train a bridge

### Prepare the inputs

Split participants into training, validation, and test sets before fitting or
pairing. Fit the reference classifier on reference-training data, then freeze
it and the encoder. Extract wrist (`source`) and reference features with the
same encoder and fitted projection. Supply features before standardization.
The training code estimates normalization from unique reference-training rows.

Each training row pairs a labeled wrist window with a reference window. Use
concurrent recordings for Thigh2Wrist, or match within participant, activity,
and wrist-side groups for Lab2Freeliving.

Save a dictionary with these keys as `calibration.pt` using `torch.save`.
Command-line training uses a fitted `torch.nn.Linear(D, C)` classifier that
returns logits. `N` is the number of pairs, `D` the feature dimension, and `C`
the number of activities.

| Key | Contents |
| --- | --- |
| `source` | Float32 tensor `[N, D]` of wrist features |
| `reference` | Float32 tensor `[N, D]` of paired reference features |
| `labels` | Int64 tensor `[N]` of wrist activity labels, from `0` to `C-1` |
| `reference_train` | Float32 tensor `[M, D]` of unique reference-training features |
| `classifier` | The fitted classifier's `state_dict()` |
| `weights` | Optional positive float32 tensor `[N]` of pair weights, defaulting to ones |

### Run training

```bash
python train.py --data calibration.pt --output bridge.pt --method meanflow
```

Choose `mapper`, `meanflow`, or `distillation` with `--method`. Add
`--device cuda` for GPU training. Run `python train.py --help` for all options.

The default run trains for 30 epochs and saves the final checkpoint. To use
early stopping and select the best validation macro-F1, add
`--validation validation.pt`. This file contains `source` and `labels` for
unique validation windows, using the tensor formats above.

## Predict activities

Extract wrist features with the same encoder, projection, preprocessing, and
class order used for training, then load the saved bridge:

```python
from train import load_bridge
from wearablebridge import predict

bridge, classifier, standardizer = load_bridge("bridge.pt")
probabilities = predict(source_features.cpu().float(), bridge, classifier, standardizer)
labels = probabilities.argmax(dim=1)
```

Process large datasets in batches.

<details>
<summary>Python API: custom classifiers and pairs that change each epoch</summary>

`train_bridge` accepts an iterable of `(source, reference, labels, weights)` batches
or a function `batches(epoch)` that returns them. Epoch numbers start at 1.
Validation batches contain `(source, labels)`.

```python
import torch
from train import train_bridge

checkpoint = train_bridge(batches, reference_train, classifier,
                          validation=validation_batches, seed=20260717)
torch.save(checkpoint, "bridge.pt")
```

Seed and recreate the loader or pairing function for each run. Weighted losses
use `mean(weight * loss)`. To restore a custom classifier, pass its architecture
to `load_bridge("bridge.pt", classifier=model)`.

</details>

<details>
<summary>Temporal averaging</summary>

For the paper's five-window results:

```python
from wearablebridge import causal_average

probabilities = causal_average(probabilities, stream, start_us, window_count=5)
labels = probabilities.argmax(dim=1)
```

`stream` identifies each participant/wrist recording stream. `start_us` contains
integer window-start times in microseconds. Include all windows before applying
the activity scoring mask. History resets at gaps.

</details>

## Settings used in the paper

The primary four-class experiments use frozen MoCA features with these settings:

| Setting | Thigh2Wrist | Lab2Freeliving |
| --- | --- | --- |
| Feature dimension | 768 | 128 |
| Frozen classifier | `Linear(768, 4)` | `Linear(128, 4)` |
| `--hidden-dim` | 768 | 256 |
| MeanFlow `--epsilon` | 0.06 | 0.01 |
| Checkpoint selection | Final epoch | Best validation macro-F1 |

Both designs use batches of 128, learning rate and weight decay of `1e-4`, and
at most 30 epochs. Paper seeds are `20260717`, `20260718`, and `20260719`.
The CLI defaults to seed 0. Set `--seed` to use a paper seed.

For Lab2Freeliving, the 128-dimensional features are the output of a fitted
linear projection, LayerNorm, and GELU. Freeze this block together with the
classifier. Rematch pairs each epoch through the Python API, using the same
pairing schedule across methods. Within each participant/activity/wrist-side
group, shuffle both lists independently, cycle the shorter list, and then
shuffle all pairs together.

See the manuscript appendix for preprocessing, reference-model fitting,
participant splits, pair weights, and evaluation.

<details>
<summary>Encoders and pretrained weights</summary>

| Encoder | Source | Weights used in the paper |
| --- | --- | --- |
| MoCA | [Ryu et al. (2025), code](https://github.com/HowonRyu/MoCA) | [checkpoint-19.pth](https://github.com/HowonRyu/MoCA/blob/main/checkpoints/checkpoint-19.pth) |
| HARNet10 | [Yuan et al. (2024), code](https://github.com/OxWearables/ssl-wearables) | `mtl_best.mdl` |
| DeepConvLSTM | Ordóñez and Roggen (2016), doi:10.3390/s16010115 | Trained from scratch |
| InceptionTime | Ismail Fawaz et al. (2020), doi:10.1007/s10618-020-00710-y | Trained from scratch |
| HART | Ek et al. (2023), doi:10.1007/s00779-023-01776-3 | Accelerometer-only adaptation, trained from scratch |

Reference classifiers and the MoCA projection are fitted on reference data.
They are separate from the downloaded pretrained weights. Obtain encoder code,
datasets, and pretrained checkpoints from their respective sources.

</details>

<details>
<summary>Try bridging on held-out synthetic data</summary>

On macOS or Linux, paste this entire block into the terminal after installation,
from the `WearableBridge` folder with the virtual environment active. It runs
on the CPU and cleans up its temporary checkpoint automatically.

The example creates four synthetic activities and adds a feature-dependent
offset to the source features. A known reference classifier recognizes the
reference features. MeanFlow learns from 256 training pairs, then the code
compares unadapted and bridged predictions on 256 new samples.

```bash
python - <<'PY'
import tempfile
from pathlib import Path
import torch
from train import train_bridge, load_bridge
from wearablebridge import predict

torch.set_num_threads(1)
torch.manual_seed(7)

def make_data(n):
    labels = torch.arange(n) % 4
    reference = 2 * torch.eye(4)[labels] + 0.2 * torch.randn(n, 4)
    source = reference + torch.tensor([2.5, -1.0, 0.5, -1.5])
    return source, reference, labels

source, reference, labels = make_data(256)
test_source, _, test_labels = make_data(256)
classifier = torch.nn.Linear(4, 4)
with torch.no_grad():
    classifier.weight.copy_(torch.eye(4))
    classifier.bias.zero_()

checkpoint = train_bridge(
    [(source, reference, labels, torch.ones(len(labels)))],
    reference, classifier, method="meanflow", hidden_dim=32,
    epochs=60, learning_rate=0.01, seed=7,
)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "bridge.pt"
    torch.save(checkpoint, path)
    bridge, head, standardizer = load_bridge(path)
    before = predict(test_source, None, head, standardizer)
    after = predict(test_source, bridge, head, standardizer)

for name, probabilities in [("Unadapted", before), ("MeanFlow bridge", after)]:
    assert probabilities.shape == (256, 4) and torch.isfinite(probabilities).all()
    accuracy = (probabilities.argmax(1) == test_labels).float().mean().item()
    print(f"{name}: {accuracy:.1%} accuracy on held-out synthetic samples")
PY
```

Verified output with the pinned dependencies on CPU:

```text
Unadapted: 37.1% accuracy on held-out synthetic samples
MeanFlow bridge: 100.0% accuracy on held-out synthetic samples
```

This controlled example illustrates correction of a feature mismatch. The
paper evaluates performance on wearable recordings and compares methods across
participants, datasets, and training runs.

</details>

## License

The bridge code and demo viewer are released under the [MIT License](LICENSE).
External encoders and datasets retain their respective licenses and access terms.
