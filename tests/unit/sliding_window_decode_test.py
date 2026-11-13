# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode with LOCAL_SLIDING attention must match a train-mode forward over the same tokens.

Single-token decode attends over the prefill cache and the AR cache separately. The sliding
window has to be measured between true token positions, which matters once the AR cache
(`max_target_length - max_prefill_predict_length`) is longer than `sliding_window_size`.
The AR write index is shared by all slots and wraps, so a request inserted while it is
past 0 fills the end of the AR cache and then its start.
"""

import sys
import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.attention_op import AttentionOp
from maxtext.utils import maxtext_utils, model_creation_utils

from maxtext.inference.maxengine import maxengine
from tests.utils.test_helpers import get_test_config_path

# In cpu-unit where jetstream is not installed, gcloud_stub's ResultTokens is a plain class;
# register it as a JAX pytree so MaxEngine._prefill_jit and _generate_jit can return it.
if "_jetstream_stubs" in getattr(maxengine.engine_api.ResultTokens, "__qualname__", ""):
  try:
    jax.tree_util.register_pytree_node(
        maxengine.engine_api.ResultTokens,
        lambda r: ((r.data, r.log_prob), (r.tokens_idx, r.valid_idx, r.length_idx, r.samples_per_slot)),
        lambda aux, children: maxengine.engine_api.ResultTokens(
            data=children[0],
            log_prob=children[1],
            tokens_idx=aux[0],
            valid_idx=aux[1],
            length_idx=aux[2],
            samples_per_slot=aux[3],
        ),
    )
  except ValueError:
    pass

PROMPT = [3, 17, 42, 5, 9]
STEPS = 30
REL_TOL = 1e-3

_COMMON = {
    "override_model_config": True,
    "base_num_decoder_layers": 2,
    "base_emb_dim": 128,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "vocab_size": 128,
    # AR cache of 48 slots is longer than the window of 8.
    "sliding_window_size": 8,
    "max_prefill_predict_length": 16,
    "max_target_length": 64,
    "per_device_batch_size": 1,
    "scan_layers": False,
    "attention": "dot_product",
    "dtype": "float32",
    "weight_dtype": "float32",
    "matmul_precision": "highest",
    "decode_sampling_strategy": "greedy",
    "enable_checkpointing": False,
    "skip_jax_distributed_system": True,
}

# Six layers give Gemma 3 and Gemma 4 their global layer after five local ones.
_REUSE_MODELS = {
    "gemma2": {"model_name": "gemma2-2b", "base_mlp_dim": 256},
    "gemma3": {"model_name": "gemma3-4b", "base_num_decoder_layers": 6, "base_mlp_dim": 256, "use_multimodal": False},
    "gemma4": {
        "model_name": "gemma4-31b",
        "base_num_decoder_layers": 6,
        "base_mlp_dim": 256,
        "global_head_dim": 64,
        "global_num_kv_heads": 1,
        "use_multimodal": False,
    },
}


class SlidingWindowDecodeTest(parameterized.TestCase):
  """Per-step decode logits must equal the forward-pass logits at the same position."""

  @parameterized.named_parameters(
      ("gemma2", {"model_name": "gemma2-2b", "base_mlp_dim": 256}),
      (
          "gpt_oss",
          {
              "model_name": "gpt-oss-20b",
              "base_mlp_dim": 128,
              "base_moe_mlp_dim": 128,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "sparse_matmul": False,
              "megablox": False,
          },
      ),
  )
  def test_decode_matches_forward_past_window(self, overrides):
    cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **_COMMON, **overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0))
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    # Default init is close to degenerate for attention; random weights make the window visible.
    leaves, treedef = jax.tree.flatten(params_state)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    leaves = [jax.random.normal(k, x.shape, x.dtype) * 0.2 for k, x in zip(keys, leaves)]
    params_state = jax.tree.unflatten(treedef, leaves)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    padded = jnp.asarray(PROMPT + [0] * (cfg.max_prefill_predict_length - len(PROMPT)), dtype=jnp.int32)
    prefix, first_token = engine.prefill(params=params, padded_tokens=padded, true_length=len(PROMPT))
    tokens = list(PROMPT) + [int(first_token.data[0, 0])]
    decode_logits = [np.asarray(prefix["logits"], np.float64).reshape(-1)[-cfg.vocab_size :]]
    state = engine.insert(prefix, engine.init_decode_state(), slot=0)
    for _ in range(STEPS):
      state, result = engine.generate(params, state)
      decode_logits.append(np.asarray(state["logits"][0], np.float64).reshape(-1)[-cfg.vocab_size :])
      tokens.append(int(result.data[0, 0]))

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
    nnx.update(train_model, params_state)
    seq, batch = cfg.max_target_length, cfg.micro_batch_size_to_train_on
    ids = jnp.tile(jnp.asarray([tokens + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    positions = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    segment_ids = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      ref = train_model(
          ids, positions, decoder_segment_ids=segment_ids, enable_dropout=False, model_mode=MODEL_MODE_TRAIN
      )
    ref = np.asarray(ref[0] if isinstance(ref, tuple) else ref, np.float64)[0]

    rels = [
        float(np.max(np.abs(d - ref[len(PROMPT) - 1 + k])) / np.max(np.abs(ref[len(PROMPT) - 1 + k])))
        for k, d in enumerate(decode_logits)
    ]
    worst = int(np.argmax(rels))
    self.assertLessEqual(rels[worst], REL_TOL, f"decode step {worst} rel max-abs {rels[worst]:.3e}")

  def test_ar_cache_positions_need_segment_ids(self):
    """Without AR cache segment ids the true-position mask defers to the slot-index fallback."""
    positions = jnp.array([[9]], dtype=jnp.int32)
    segment_ids = jnp.array([[1, 1, 1, 0]], dtype=jnp.int32)
    # pylint: disable=protected-access
    self.assertIsNone(AttentionOp._decode_cache_positions(None, positions, None, 4, "ar"))
    abs_q, abs_k = AttentionOp._decode_cache_positions(None, positions, segment_ids, 4, "ar")
    # The newest valid AR slot (2) holds the query token at position 9.
    np.testing.assert_array_equal(np.asarray(abs_q - abs_k)[0, 0], [2, 1, 0, -1])

  def test_ar_cache_positions_after_shared_index_wraps(self):
    """Valid AR slots that wrap past the end of the cache end at the newest token, not at the last slot."""
    positions = jnp.array([[9], [9], [9]], dtype=jnp.int32)
    segment_ids = jnp.array(
        [
            # Inserted at index 3, three tokens written at 3, 0 and 1.
            [1, 1, 0, 1],
            # Inserted at index 1, not wrapped.
            [0, 1, 1, 0],
            # Full: no run end, so the highest valid slot stays the newest.
            [1, 1, 1, 1],
        ],
        dtype=jnp.int32,
    )
    # pylint: disable=protected-access
    abs_q, abs_k = AttentionOp._decode_cache_positions(None, positions, segment_ids, 4, "ar")
    np.testing.assert_array_equal(
        np.asarray(abs_q - abs_k)[:, 0],
        [[1, 0, -1, 2], [2, 1, 0, -1], [3, 2, 1, 0]],
    )

  @parameterized.named_parameters(
      ("gemma2_ar_cache_longer_than_window", "gemma2", 24),
      ("gemma3_ar_cache_longer_than_window", "gemma3", 24),
      ("gemma4_ar_cache_longer_than_window", "gemma4", 24),
      ("gemma4_ar_cache_within_window", "gemma4", 8),
  )
  def test_decode_matches_forward_after_shared_ar_index_wraps(self, family, ar_cache):
    """A request inserted into a second slot while the shared AR index is halfway decodes past the wrap.

    Slot 0 decodes `ar_cache // 2` steps first, so the request in slot 1 starts at that AR index and
    wraps after `ar_cache - ar_cache // 2` steps. It decodes `ar_cache - 1` steps, the most a request
    can: prefill gives the first of at most `ar_cache` tokens.
    """
    overrides = {
        **_COMMON,
        **_REUSE_MODELS[family],
        "max_target_length": _COMMON["max_prefill_predict_length"] + ar_cache,
        "per_device_batch_size": 2,
    }
    cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0))
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    leaves, treedef = jax.tree.flatten(params_state)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    leaves = [jax.random.normal(k, x.shape, x.dtype) * 0.2 for k, x in zip(keys, leaves)]
    params_state = jax.tree.unflatten(treedef, leaves)

    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)

    def prefill(prompt):
      padded = jnp.asarray(prompt + [0] * (cfg.max_prefill_predict_length - len(prompt)), dtype=jnp.int32)
      return engine.prefill(params=params, padded_tokens=padded, true_length=len(prompt))

    first_prefix, _ = prefill([11, 23, 7, 61, 2])
    state = engine.insert(first_prefix, engine.init_decode_state(), slot=0)
    for _ in range(ar_cache // 2):
      state, _ = engine.generate(params, state)
    prefix, first_token = prefill(PROMPT)
    tokens = list(PROMPT) + [int(first_token.data[0, 0])]
    decode_logits = [np.asarray(prefix["logits"], np.float64).reshape(-1)[-cfg.vocab_size :]]
    state = engine.insert(prefix, state, slot=1)
    for _ in range(ar_cache - 1):
      state, result = engine.generate(params, state)
      decode_logits.append(np.asarray(state["logits"][1], np.float64).reshape(-1)[-cfg.vocab_size :])
      tokens.append(int(result.data[1, 0]))

    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
    nnx.update(train_model, params_state)
    seq, batch = cfg.max_target_length, cfg.micro_batch_size_to_train_on
    ids = jnp.tile(jnp.asarray([tokens + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    positions = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    segment_ids = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
      ref = train_model(
          ids, positions, decoder_segment_ids=segment_ids, enable_dropout=False, model_mode=MODEL_MODE_TRAIN
      )
    ref = np.asarray(ref[0] if isinstance(ref, tuple) else ref, np.float64)[0]

    rels = [
        float(np.max(np.abs(d - ref[len(PROMPT) - 1 + k])) / np.max(np.abs(ref[len(PROMPT) - 1 + k])))
        for k, d in enumerate(decode_logits)
    ]
    worst = int(np.argmax(rels))
    self.assertLessEqual(rels[worst], REL_TOL, f"decode step {worst} rel max-abs {rels[worst]:.3e}")


if __name__ == "__main__":
  unittest.main()
