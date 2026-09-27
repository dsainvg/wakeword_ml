"""
Streaming Keyword Spotting (KWS) Engine in Python
-------------------------------------------------
Simulates real-time on-device keyword spotting on audio streams:
- Sliding 1.0s audio buffer (16,000 samples @ 16kHz) with 100ms stride
- On-the-fly 40-band Log-Mel feature extraction
- Fast JAX inference with pre-loaded weights
- 2-step moving average confidence debouncer (suppresses spurious false alarms)
- Real-time latency tracking
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from features import AudioFeatureExtractor
from models import get_model, count_params


class StreamingKWSEngine:
    def __init__(self, checkpoint_path: str = "best_kws_model.flax", arch: str = "bcresnet",
                 confidence_threshold: float = None, ema_alpha: float = 0.60,
                 consecutive_required: int = 2, refractory_steps: int = 15,
                 aggregation: str = "maxhold", hold_decay: float = 0.85):
        """confidence_threshold=None -> use the operating point stored in the
        checkpoint (v2+ trainers store the threshold that meets the FPR budget).

        aggregation='maxhold' is the default on purpose. An EMA with alpha=0.6 needs
        roughly five consecutive high frames to cross the threshold, but a 0.6 s
        wake word only produces two or three good frames in a 1 s sliding window, so
        an EMA-confirmed detector discards most real detections. A decaying peak hold
        still requires `consecutive_required` frames above threshold, so isolated
        spikes are rejected just as well, but it confirms on a realistic burst.
        """
        self.extractor = AudioFeatureExtractor()
        self.arch = arch
        self.ema_alpha = ema_alpha
        self.consecutive_required = consecutive_required
        self.refractory_steps = refractory_steps
        self.aggregation = aggregation
        self.hold_decay = hold_decay

        # Reconstruct architecture template
        self.model = get_model(self.arch, num_classes=2)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 49, 40, 1), dtype=jnp.float32)
        variables = self.model.init(rng, dummy_input, train=False)
        dummy_params = variables['params']

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                encoded = f.read()

            template = {
                "params": dummy_params,
                "arch": arch,
                "param_count": 0,
                "val_acc": 0.0,
                "tpr": 0.0,
                "fpr": 0.0,
                "threshold": 0.85
            }
            # v1 checkpoints carry a fixed key set; v2+ add their own fields, so try
            # the fixed template first and fall back to a free-form restore.
            try:
                state_dict = serialization.from_bytes(template, encoded)
            except ValueError:
                state_dict = serialization.msgpack_restore(encoded)
            # msgpack_restore / from_bytes hand back numpy leaves. The streaming
            # predictor is jitted, and a numpy-style index inside the model would then try
            # to convert a tracer, so promote the params to jnp arrays once at load time.
            self.params = jax.tree_util.tree_map(jnp.asarray, state_dict["params"])
            self.val_acc = state_dict.get("val_acc", 1.0)
            stored_thr = state_dict.get("threshold", 0.85)
            self.confidence_threshold = float(
                stored_thr if confidence_threshold is None else confidence_threshold)
            if aggregation == "maxhold" and "deploy_agg" in state_dict:
                aggregation = state_dict["deploy_agg"]
            if "deploy_refractory_steps" in state_dict:
                self.refractory_steps = int(state_dict["deploy_refractory_steps"])
            if "deploy_need" in state_dict:
                self.consecutive_required = int(state_dict["deploy_need"])
            self.aggregation = aggregation
            self.param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))

            print("=====================================================================")
            print(f" Loaded KWS Engine: {self.arch.upper()} from {checkpoint_path}")
            print(f" Model Parameters : {self.param_count:,} (INT8 Footprint: ~{self.param_count / 1024:.2f} KB)")
            print(f" Detection Thresh : {self.confidence_threshold:.4f}"
                  f"{'  (from checkpoint)' if confidence_threshold is None else '  (override)'}")
            print(f" Confirmation     : {self.aggregation} {self.consecutive_required} hits, "
                  f"{self.refractory_steps * 0.1:.1f}s refractory")
            if "tpr_offset_stress" in state_dict:
                print(f" Stored TPR       : {state_dict['tpr_offset_stress'] * 100:.2f}% "
                      f"@ FPR <= {state_dict.get('fpr_budget', 0) * 100:.2f}%")
            if "deploy_recall" in state_dict:
                print(f" Stored streaming : recall {state_dict['deploy_recall']:.1f}%  "
                      f"FA {state_dict['deploy_fa_per_hour']:.2f}/hour")
            print("=====================================================================")
        else:
            print(f"[Warning] Checkpoint '{checkpoint_path}' not found. Using initialized weights.")
            self.params = dummy_params
            self.param_count = count_params(variables)
            self.confidence_threshold = 0.85 if confidence_threshold is None else confidence_threshold

        # JIT-compile the inference function
        @jax.jit
        def predict_fn(params, x):
            logits = self.model.apply({'params': params}, x, train=False)
            probs = jax.nn.softmax(logits, axis=-1)
            return probs[0, 1]  # Return target keyword probability

        self.predict_fn = predict_fn

        # Warmup JIT compilation
        _ = self.predict_fn(self.params, dummy_input)

        # Audio sliding window (1.0 second = 16,000 samples)
        self.buffer_len = 16000
        self.audio_buffer = np.zeros(self.buffer_len, dtype=np.float32)

        # Debouncing filter state
        self.ema = 0.0
        self.hold = 0.0
        self.consecutive_hits = 0
        self.lockout = 0

    def _aggregate(self, raw_prob: float) -> float:
        if self.aggregation == "maxhold":
            self.hold = max(raw_prob, self.hold * self.hold_decay)
            return self.hold
        self.ema = self.ema_alpha * raw_prob + (1.0 - self.ema_alpha) * self.ema
        return self.ema

    def push_audio_chunk(self, chunk: np.ndarray) -> tuple[bool, float, float]:
        """
        Feeds an incoming audio chunk (e.g. 50ms = 800 samples or 100ms = 1600 samples).
        Returns:
            (is_keyword_spotted, aggregated_confidence, inference_latency_ms)
        """
        chunk = np.array(chunk, dtype=np.float32)
        if np.max(np.abs(chunk)) > 1.0:
            chunk = chunk / 32768.0

        n = len(chunk)
        # Shift buffer left by n, append new chunk
        self.audio_buffer = np.roll(self.audio_buffer, -n)
        self.audio_buffer[-n:] = chunk

        # Feature extraction
        t0 = time.perf_counter()
        spec = self.extractor.compute_spectrogram(self.audio_buffer)
        tensor = jnp.expand_dims(jnp.expand_dims(jnp.array(spec), axis=0), axis=-1)

        # Model forward inference
        raw_prob = float(self.predict_fn(self.params, tensor))
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        score = self._aggregate(raw_prob)

        if self.lockout > 0:
            self.lockout -= 1

        # Confirmation: N consecutive windows above threshold, then refractory lockout
        keyword_detected = False
        if score >= self.confidence_threshold:
            self.consecutive_hits += 1
            if self.consecutive_hits >= self.consecutive_required and self.lockout == 0:
                keyword_detected = True
                self.consecutive_hits = 0
                self.lockout = self.refractory_steps
                self.ema = 0.0
                self.hold = 0.0
        else:
            self.consecutive_hits = 0

        return keyword_detected, score, latency_ms

    def reset(self):
        """Clears audio buffer and detection state."""
        self.audio_buffer.fill(0)
        self.ema = 0.0
        self.hold = 0.0
        self.consecutive_hits = 0
        self.lockout = 0


if __name__ == "__main__":
    engine = StreamingKWSEngine(checkpoint_path="best_kws_model.flax", arch="bcresnet")
    print("\nSimulating streaming silence...")
    for _ in range(5):
        chunk = np.random.normal(0, 0.01, 1600).astype(np.float32)
        spotted, conf, lat = engine.push_audio_chunk(chunk)
        print(f"Confidence: {conf:.3f} | Spotted: {spotted} | Latency: {lat:.2f} ms")
