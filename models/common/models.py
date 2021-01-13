from typing import List, Tuple, Iterable, NamedTuple

import tensorflow as tf


class FullyConnected(tf.Module):
    def __init__(self, layers: List[int] = None, activation=tf.tanh, layer_norm=False, activation_out=True,
                 name="fully_connected_net"):
        super().__init__(name)
        self.layers = [tf.keras.layers.Flatten()]
        with self.name_scope:
            for i, num_hidden in enumerate(layers):
                self.layers.append(tf.keras.layers.Dense(units=num_hidden, kernel_initializer='lecun_normal'))
                if layer_norm:
                    self.layers.append(tf.keras.layers.LayerNormalization())
                if activation_out or i != len(layers) - 1:
                    self.layers.append(activation)

    def __call__(self, inputs):
        fc_out = inputs
        for layer in self.layers:
            fc_out = layer(fc_out)
        return fc_out


class NatureCNN(tf.Module):

    def __init__(self, activation_out=True, name="nature_cnn_net"):
        super().__init__(name)
        with self.name_scope:
            self.layers = [
                tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4),
                tf.nn.relu,
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2),
                tf.nn.relu,
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=1),
                tf.nn.relu,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=512),
            ]
            if activation_out:
                self.layers.append(tf.nn.relu)

    def __call__(self, inputs):
        conv_out = inputs
        for layer in self.layers:
            conv_out = layer(conv_out)
        return conv_out


class ImpalaResidualLayer(NamedTuple):
    conv_in: tf.keras.layers.Conv2D
    pool: tf.keras.layers.MaxPool2D
    residual_blocks: List


class ImpalaCNN(tf.Module):

    def __init__(self, residual_layers: Iterable[Tuple[int, int]] = None, activation_out=True, name="impala_cnn_net"):
        super().__init__(name)
        residual_layers = residual_layers or [(16, 2), (32, 2), (32, 2)]
        self._residual_layers = []
        with self.name_scope:
            for num_ch, num_blocks in residual_layers:
                self._residual_layers.append(ImpalaResidualLayer(
                    conv_in=tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=1, padding='same'),
                    pool=tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
                    residual_blocks=[[
                        tf.nn.relu, tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=1, padding='same'),
                        tf.nn.relu, tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=1, padding='same')
                    ] for _ in range(num_blocks)]
                ))
            self.dense_layers = [
                tf.nn.relu,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256),
            ]
            if activation_out:
                self.dense_layers.append(tf.nn.relu)

    def __call__(self, inputs):
        conv_out = inputs
        for res_layer in self._residual_layers:
            conv_out = res_layer.conv_in(conv_out)
            conv_out = res_layer.pool(conv_out)
            for res_block in res_layer.residual_blocks:
                block_input = conv_out
                for layer in res_block:
                    conv_out = layer(conv_out)
                conv_out += block_input
        for layer in self.dense_layers:
            conv_out = layer(conv_out)
        return conv_out
