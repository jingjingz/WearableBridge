"""WearableBridge models and objectives for frozen reference features."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Standardizer(nn.Module):
    """Fit once on unique reference-training features, before pairing."""

    def __init__(self, reference_train=None, *, mean=None, scale=None):
        super().__init__()
        if reference_train is not None:
            if mean is not None or scale is not None:
                raise ValueError("provide training features or saved statistics")
            if reference_train.ndim != 2 or len(reference_train) == 0:
                raise ValueError("reference_train must be a nonempty [N, D] matrix")
            reference_train = reference_train.detach()
            mean = reference_train.mean(0)
            scale = reference_train.std(0, unbiased=False).clamp_min(1e-4)
        if mean is None or scale is None or mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("mean and scale must be equal-length vectors")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("normalization statistics must be finite")
        if not (scale > 0).all():
            raise ValueError("normalization scales must be positive")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())

    def normalize(self, features):
        return (features - self.mean) / self.scale

    def denormalize(self, features):
        return features * self.scale + self.mean


def _mlp(input_dim, hidden_dim, output_dim):
    if min(input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError("network dimensions must be positive")
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
    ]
    output = nn.Linear(hidden_dim, output_dim)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


class ResidualBridge(nn.Module):
    """Mapper and Distillation share this identity-initialized network."""

    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()
        self.network = _mlp(feature_dim, hidden_dim, feature_dim)

    def forward(self, source):
        return source + self.network(source)


def _time_embedding(time, dim):
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = time[:, None] * frequencies[None] * (2 * math.pi)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class MeanFlow(nn.Module):
    """Condition interval velocity on the observed source feature."""

    def __init__(self, feature_dim=128, hidden_dim=256, time_dim=32):
        super().__init__()
        if time_dim <= 0 or time_dim % 2:
            raise ValueError("time_dim must be positive and even")
        self.time_dim = time_dim
        self.network = _mlp(2 * feature_dim + 3 * time_dim, hidden_dim, feature_dim)

    def velocity(self, state, t0, t, source):
        times = [_time_embedding(x, self.time_dim) for x in (t0, t, t - t0)]
        return self.network(torch.cat((state, source, *times), dim=1))

    def forward(self, source):
        zero = source.new_zeros(len(source))
        one = source.new_ones(len(source))
        return source - self.velocity(source, zero, one, source)


def sample_times(source, generator):
    """Draw logit-normal intervals [t0, t]; set half of the batch to t0 = t."""

    batch = len(source)
    times = torch.randn(
        (2, batch), device=source.device, dtype=source.dtype, generator=generator
    )
    times = torch.sigmoid(times - 0.4)
    t0, t = times.min(0).values, times.max(0).values
    boundary = torch.randperm(batch, device=source.device, generator=generator)[
        : round(batch / 2)
    ]
    t0[boundary] = t[boundary]
    return t0, t


def meanflow_loss(model, source, target, t0, t, epsilon=0.01):
    """Return one adaptive Improved MeanFlow loss per calibration pair."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    state = (1 - t[:, None]) * target + t[:, None] * source
    instantaneous = model.velocity(state, t, t, source)
    average, derivative = torch.func.jvp(
        model.velocity,
        (state, t0, t, source),
        (
            instantaneous,
            torch.zeros_like(t0),
            torch.ones_like(t),
            torch.zeros_like(source),
        ),
    )
    # Hold t0 and the source condition fixed; stop gradients through the correction.
    prediction = average + (t - t0)[:, None] * derivative.detach()
    error = (prediction - (source - target)).square().sum(1)
    return error / (error.detach() + epsilon)


def bridge_loss(
    method,
    bridge,
    classifier,
    standardizer,
    source_features,
    reference_features,
    labels,
    weights,
    *,
    generator=None,
    epsilon=0.01,
):
    """Compute the batch mean of weighted pair losses for a frozen classifier."""

    if method not in ("mapper", "meanflow", "distillation"):
        raise ValueError(f"unknown method: {method}")
    if classifier.training or any(p.requires_grad for p in classifier.parameters()):
        raise ValueError("freeze the classifier with .eval().requires_grad_(False)")
    if source_features.ndim != 2 or source_features.shape != reference_features.shape:
        raise ValueError("source and reference features must have equal [N, D] shapes")
    if len(source_features) == 0 or labels.shape != (len(source_features),):
        raise ValueError("labels must align with a nonempty batch")
    if weights.shape != labels.shape or not torch.isfinite(weights).all():
        raise ValueError("weights must be finite and aligned with labels")
    if not (weights > 0).all():
        raise ValueError("weights must be positive")

    source = standardizer.normalize(source_features.detach())
    target = standardizer.normalize(reference_features.detach())
    generated = bridge(source)
    logits = classifier(standardizer.denormalize(generated))
    loss = F.cross_entropy(logits, labels, reduction="none")
    if method == "mapper":
        loss = loss + (generated - target).square().mean(1)
    elif method == "distillation":
        temperature = 2.0
        with torch.no_grad():
            teacher = F.softmax(classifier(reference_features) / temperature, dim=-1)
        divergence = F.kl_div(
            F.log_softmax(logits / temperature, dim=-1), teacher, reduction="none"
        ).sum(1)
        loss = loss + temperature**2 * divergence
    else:
        if generator is None:
            raise ValueError("MeanFlow requires a seeded torch.Generator")
        t0, t = sample_times(source, generator)
        loss = loss + meanflow_loss(bridge, source, target, t0, t, epsilon)
        loss = loss + 0.01 * (generated - source).square().mean(1)
    if not torch.isfinite(loss).all():
        raise ValueError("nonfinite bridge loss")
    # Weights use full training-set counts and are not renormalized within batches.
    return (loss * weights.to(loss)).mean()


@torch.no_grad()
def predict(source_features, bridge, classifier, standardizer):
    """Return class probabilities; bridge=None gives the unadapted baseline."""

    classifier.eval()
    if bridge is None:
        return classifier(source_features).softmax(-1)
    bridge.eval()
    source = standardizer.normalize(source_features)
    features = standardizer.denormalize(bridge(source))
    return classifier(features).softmax(-1)


@torch.no_grad()
def causal_average(probabilities, stream, start_us, window_count=5):
    """Average contiguous ten-second windows within each participant/wrist stream.

    Include all available predictions, then apply the activity scoring mask.
    The result is a CPU float32 tensor in the original row order.
    """

    count = len(probabilities)
    if probabilities.ndim != 2 or count == 0:
        raise ValueError("probabilities must be a nonempty [N, classes] matrix")
    if stream.shape != (count,) or start_us.shape != (count,):
        raise ValueError("stream and start_us must align with probabilities")
    if window_count < 1 or not torch.isfinite(probabilities).all():
        raise ValueError("use finite probabilities and a positive window_count")
    probabilities = probabilities.detach().to(device="cpu", dtype=torch.float32)
    stream = stream.to(device="cpu", dtype=torch.long)
    start_us = start_us.to(device="cpu", dtype=torch.long)
    by_time = torch.argsort(start_us, stable=True)
    order = by_time[torch.argsort(stream[by_time], stable=True)]
    same_stream = stream[order][1:] == stream[order][:-1]
    time_step = start_us[order][1:] - start_us[order][:-1]
    if (same_stream & (time_step == 0)).any():
        raise ValueError("each stream must have unique window timestamps")
    new_run = torch.ones(count, dtype=torch.bool)
    new_run[1:] = ~same_stream | (time_step != 10_000_000)
    starts = new_run.nonzero().flatten()
    ends = torch.cat((starts[1:], torch.tensor([count])))
    ordered = probabilities[order]
    output = torch.empty_like(ordered)
    for first, stop in zip(starts.tolist(), ends.tolist()):
        run = ordered[first:stop]
        total = run.clone()
        for lag in range(1, min(window_count, len(run))):
            total[lag:] += run[:-lag]
        divisor = torch.arange(1, len(run) + 1).clamp_max(window_count)
        output[order[first:stop]] = total / divisor[:, None]
    return output
