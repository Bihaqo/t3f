# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A copy of tf.test.benchmark_config() to be used until next stable release.
Copied with minor modifications from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/benchmark.py
"""
import tensorflow.compat.v1 as tf


def import_benchmark_config():
  try:
    tf.test.benchmark_config()
  except AttributeError:
    from tensorflow import core
    def benchmark_config():
      """Returns a tf.ConfigProto for disabling the dependency optimizer.
        Returns:
          A TensorFlow ConfigProto object.
      """
      config = core.protobuf.config_pb2.ConfigProto()
      config.graph_options.rewrite_options.dependency_optimization = (
        core.protobuf.rewriter_config_pb2.RewriterConfig.OFF)
      return config
    tf.test.benchmark_config = benchmark_config
