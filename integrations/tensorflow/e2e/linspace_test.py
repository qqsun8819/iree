# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class LinSpaceModule(tf.Module):

  def __init__(self):
    pass

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.int32)
  ])
  def linspace(self, start, stop, num):
    return tf.linspace(start, stop, num)


# TODO(jennik): Get this test working on IREE by implementing the linspace op in
# MLIR.
@tf_test_utils.compile_modules(backends=["tf"], linspace=LinSpaceModule)
class LinspaceTest(tf_test_utils.SavedModelTestCase):

  def test_linspace(self):
    start = np.array(10., dtype=np.float32)
    stop = np.array(12., dtype=np.float32)
    num = np.array(3, dtype=np.int32)

    result = self.modules.linspace.all.linspace(start, stop, num)
    result.assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
