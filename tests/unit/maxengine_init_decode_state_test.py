# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MaxEngine.init_decode_state on the NNX path (no JetStream or checkpoint needed)."""

import gc
import sys
import unittest

import jax

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from tests.utils.test_helpers import get_test_config_path


class InitDecodeStateNNXTest(unittest.TestCase):
  """init_decode_state must make the decode cache only inside its sharded jit."""

  def test_init_decode_state_keeps_no_unsharded_copy_of_the_cache(self):
    """Only the arrays of the returned decode state may stay alive.

    Concrete zeros made before the jit put the whole cache on one device before the jit made it again with the
    cache shardings, and the jit's closure kept that copy alive. A cache larger than one device's memory then
    failed with RESOURCE_EXHAUSTED.
    """
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        base_emb_dim=32,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        base_num_decoder_layers=2,
        max_prefill_predict_length=4,
        max_target_length=8,
        per_device_batch_size=1,
        enable_checkpointing=False,
    )
    engine = maxengine.MaxEngine(cfg, jax.devices())
    # init_decode_state needs the kv-cache shardings but no parameters, so no checkpoint is loaded.
    engine.state_mesh_annotations = None
    engine.abstract_params = None
    maxengine.set_engine_vars_from_base_engine(engine, engine, jax.random.PRNGKey(0))

    # Holding the arrays keeps their ids from being reused by arrays made during the call.
    live_before = jax.live_arrays()
    decode_state = engine.init_decode_state()
    gc.collect()

    known = {id(x) for x in live_before} | {id(x) for x in jax.tree_util.tree_leaves(decode_state)}
    left_over = [(x.shape, x.dtype) for x in jax.live_arrays() if id(x) not in known]
    self.assertEqual(left_over, [])


if __name__ == "__main__":
  unittest.main()
