# coding=utf-8
# Copyright (c) 2021, Arm Limited and Contributors.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keyword-Transformer model."""


from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils

from kws_streaming.models.transformer_utils import KWSTransformer

import tensorflow_addons as tfa

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Permute,
    Concatenate
)

from tensorflow.keras.initializers import TruncatedNormal, Zeros

TRUNC_STD = 0.02


def model_parameters(parser_nn):
  """Keyword-Transformer model parameters."""
  parser_nn.add_argument(
      '--num_layers',
      type=int,
      default=8,
      help='The number of transformer layers',
  )
  parser_nn.add_argument(
      '--d_model',
      type=int,
      default=128,
      help='Transformer embedding dimension',
  )
  parser_nn.add_argument(
      '--mlp_dim',
      type=int,
      default=512,
      help='Transformer MLP dimension',
  )
  parser_nn.add_argument(
      '--heads',
      type=int,
      default=4,
      help='Number of heads in multihead attention',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.1,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--attention_type',
      type=str,
      default='time',
      help='Domain for attention: time, freq, both or patch',
  )
  parser_nn.add_argument(
      '--patch_size',
      type=str,
      default='1,40',
      help='Patch size in units (time steps, frequency steps)',
  )
  parser_nn.add_argument(
      '--prenorm',
      type=int,
      default=False,
      help='If True, use prenorm instead of postnorm (default: postnorm)',
  )
  parser_nn.add_argument(
      '--approximate_gelu',
      type=int,
      default=False,
      help='If True, use approximate GELU activation (useful for TFLite conversion)')


def extract_patches(images, patch_size_t, patch_size_f):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size_t, patch_size_f, 1],
        strides=[1, patch_size_t, patch_size_f, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches = tf.reshape(patches, [batch_size, -1, patch_size_f * patch_size_t])
    return patches


def model(flags):
  """ Fully attentional KWS model consisting of sequential transformer blocks.
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  distill_token = True if flags.distill_teacher_json else False

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  _, num_time_windows, num_freqs = net.shape

  if flags.attention_type == 'patch':
     patch_size_t, patch_size_f = utils.parse(flags.patch_size)
     num_patches = (num_time_windows // patch_size_t) * (num_freqs // patch_size_f)
     net = tf.expand_dims(net, axis=-1)
     patch_transformer = KWSTransformer(num_layers=flags.num_layers,
        num_classes=flags.label_count,
        d_model=flags.d_model,
        num_heads=flags.heads,
        mlp_dim=flags.mlp_dim,
        dropout=flags.dropout1,
        num_patches=num_patches,
        prenorm=flags.prenorm,
        distill_token=distill_token,
        approximate_gelu=flags.approximate_gelu,
        )

     patch_sig = extract_patches(net, patch_size_t, patch_size_f)
     patch_sig = patch_transformer(patch_sig, training=flags.training)

  if flags.attention_type == 'time' or flags.attention_type == 'both':
    time_transformer = KWSTransformer(num_layers=flags.num_layers,
        num_classes=flags.label_count,
        d_model=flags.d_model,
        num_heads=flags.heads,
        mlp_dim=flags.mlp_dim,
        dropout=flags.dropout1,
        num_patches=num_time_windows,
        prenorm=flags.prenorm,
        distill_token=distill_token,
        approximate_gelu=flags.approximate_gelu,
        )

    time_sig = time_transformer(net, training=flags.training)

  if flags.attention_type == 'freq' or flags.attention_type == 'both':
    freq_transformer = KWSTransformer(num_layers=flags.num_layers,
        num_classes=flags.label_count,
        d_model=flags.d_model,
        num_heads=flags.heads,
        mlp_dim=flags.mlp_dim,
        dropout=flags.dropout1,
        num_patches=num_freqs,
        prenorm=flags.prenorm,
        distill_token=distill_token,
        approximate_gelu=flags.approximate_gelu
        )

    net = Permute((2, 1))(net)
    freq_sig = freq_transformer(net, training=flags.training)

  mlp_heads = [ tf.keras.Sequential(
      [
          Dense(flags.label_count, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros()),
      ])
      for _ in range(2 if distill_token else 1) ]

  if flags.attention_type == 'time':
    net = time_sig
  elif flags.attention_type == 'freq':
    net = freq_sig
  elif flags.attention_type == 'both':
    net = Concatenate(axis=-1)([time_sig, freq_sig])
  elif flags.attention_type == 'patch':
    net = patch_sig
  else:
    raise ValueError('Unsupported attention type:%s' % flags.attention_type)

  if distill_token:
    heads = net
  else:
    heads = [ net ]

  outputs = []
  for mlp, head in zip(mlp_heads, heads):
    out = mlp(head)

    if flags.return_softmax:
      out = tf.keras.layers.Activation('softmax')(out)
    outputs.append(out)

  if distill_token: # Add a special output that is the average of all the other outputs
    average_output = tf.stop_gradient(sum(out for out in outputs) / len(outputs))
    outputs.append(average_output)

  return tf.keras.Model(input_audio, outputs)
