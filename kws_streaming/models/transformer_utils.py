# coding=utf-8
# Copyright (c) 2021, Arm Limited and Contributors.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vision transformer implementation based on https://github.com/tuvovan/Vision_Transformer_Keras/blob/master/vit.py

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.initializers import Zeros, Ones, TruncatedNormal, Constant
from tensorflow.python.ops import math_ops

TRUNC_STD = 0.02


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
        self.key_dense = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
        self.value_dense = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
        self.combine_heads = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros())


    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights


    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output, weights

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, prenorm=False, approximate_gelu=False):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros()),
                tfa.layers.GELU(approximate=approximate_gelu),
                Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros()),
            ]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.prenorm = prenorm


    def call(self, inputs, training):
        if self.prenorm:
            x = self.layernorm1(inputs)
            x, weights = self.att(x)
            x = self.dropout1(x, training=training)
            x = x + inputs
            y = self.layernorm2(x)
            y = self.ffn(y)
            output = x + self.dropout2(y, training=training)
        else:
            attn_output, weights = self.att(inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            output = self.layernorm2(out1 + ffn_output)
        return output, weights


class KWSTransformer(tf.keras.Model):
    def __init__(
        self,
        num_patches,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
        prenorm=False,
        distill_token=False,
        approximate_gelu=False,
    ):
        super(KWSTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        additional_tokens = 2 if distill_token else 1
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + additional_tokens, d_model), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD))
        self.distill_emb = self.add_weight("distill_emb", shape=(1, 1, d_model), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD)) if distill_token else None
        self.patch_proj = Dense(d_model, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros(), input_shape=(98,40,))

        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout, prenorm, approximate_gelu)
            for _ in range(num_layers)
        ]


    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches


    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.patch_proj(x)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        if self.distill_emb is not None:
            distill_emb = tf.broadcast_to(
                self.distill_emb, [batch_size, 1, self.d_model]
            )
            tokens = [class_emb, distill_emb, x]
        else:
            tokens = [class_emb, x]

        x = tf.concat(tokens, axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x, _ = layer(x, training)

        # First (class token) is used for classification, second for distillation (if enabled)
        class_output = x[:, 0]
        if self.distill_emb is not None:
            distill_output = x[:, 1]
            return class_output, distill_output
        else:
            return class_output
