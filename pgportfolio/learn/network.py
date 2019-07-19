#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import tensorflow as tf
import tflearn
from pgportfolio.learn.tcn import TemporalConvNet as tcn
import logging

class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, layers, device, config):
        tf_config = tf.ConfigProto()
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2 # Why? We don't run anything in parallel. :/
        self.input_num = tf.placeholder(tf.int32, shape=[])
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns])
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows])
        self.consumptions_vector = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows = rows
        self._columns = columns

        self.layers_dict = {}
        self.layer_count = 0

        self.output = self._build_network(layers)

        self.config = config

    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows (no of coins), columns (window len))
    def __init__(self, feature_number, rows, columns, layers, device, consumption_vector, config):

        self.config = config

        if self.config["training"]["consumption_scaling"] == "const" or self.config["training"]["consumption_scaling"] == "biased":
            ncv = np.ones([len (consumption_vector)]) / consumption_vector
        elif self.config["training"]["consumption_scaling"] == "linear":
            ncv = 1. / np.array (consumption_vector)
        elif self.config["training"]["consumption_scaling"] == "sqrt":
            ncv = 1. / np.sqrt (consumption_vector)        # <--- use this!
        elif self.config["training"]["consumption_scaling"] == "sqrtsqrt":
            ncv = 1. / np.sqrt (np.sqrt (consumption_vector)) # <--- This might be better!
        elif self.config["training"]["consumption_scaling"] == "s3":
            ncv = 1. / np.sqrt (np.sqrt (np.sqrt (consumption_vector)))
        else:
            assert False, 'Unrecognized consumption scaling method: ' + self.config["training"]["consumption_scaling"]

        self._ncv = ncv / np.mean (ncv)
        logging.error ("Normalized consumptions vector -- " + str(self._ncv))
#        ct = np.ones ([feature_number, rows, columns]) * ncv
        ct = np.ones ([feature_number, columns, rows]) * self._ncv    # features, coins, window # ??? How do you make sure it is broadcast in the second dimension?
        logging.error ("Consumptions tensor -- " + str(ct))
        logging.error ("Consumptions tensor shape -- " + str(ct.shape))
        ctt = np.transpose (ct, (2, 1, 0))                      
        logging.error ("Transposed consumptions tensor -- " + str(ctt))
        self.ct = tf.constant (ctt, tf.float32)
#        concat_ct = np.ones([1, 1, rows]) * self._ncv # 11 is from number of conv filters? columns=31 (window size). Is rows the number of coins?
#        cctv = np.transpose (concat_ct, (2, 1, 0))
#        self.cctv = tf.constant (cctv, tf.float32)
#        self.ct = tf.ones ([feature_number, rows, columns]) * tf_cv
#        tf_cv = tf.constant (consumption_vector / np.mean (consumption_vector))
#        self.ct = tf.ones ([feature_number, rows, columns]) * tf_cv
#        logging.error ("Consumptions Tensor -- " + str(tf_cv))

        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, device, config)

    def add_layer_to_dict(self, layer_type, tensor, weights=True):

        self.layers_dict[layer_type + '_' + str(self.layer_count) + '_activation'] = tensor
        self.layer_count += 1

    # grenrate the operation, the forward computaion
    def _build_network(self, layers):
        network = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        # [batch, assets, window, features]
        network = network / network[:, :, -1, 0, None, None]
        if self.config["training"]["consumption_scaling"] != "biased":
            network = network - 1
            network = network * self.ct
#        network = network + 1  # Performs better without this part. Better to have inputs 0-centered, no?
        tflearn.config.init_training_mode()
        for layer_number, layer in enumerate(layers):
            if layer["type"] == "DenseLayer":
                network = tflearn.layers.core.fully_connected(network,
                                                              int(layer["neuron_number"]),
                                                              layer["activation_function"],
                                                              regularizer=layer["regularizer"],
                                                              weight_decay=layer["weight_decay"] )
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "DropOut":
                network = tflearn.layers.core.dropout(network, layer["keep_probability"])
            elif layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                logging.error('network shape before EIIE_Dense: ' + str(network.get_shape()));
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 [1, width],
                                                 [1, 1],
                                                 "valid",
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                logging.error('network shape after EIIE_Dense: ' + str(network.get_shape()));
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "ConvLayer":
                logging.error('Shape before ConvLayer: ' + str(network.get_shape()))
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 allint(layer["filter_shape"]),
                                                 allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                logging.error('Shape after ConvLayer: ' + str(network.get_shape()))
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "TCNLayer":
#                network = tcn (network, int(layer["filter_number"]), 0)
                logging.error('Before reshape, TCN input is of shape: ' + str(network.get_shape()))
                orig_shape = network.get_shape()
#                orig_shape[0] = -1
#                tf.reshape (network, [-1, network.get_shape()[2], network.get_shape()[3]])
#                network = tf.reshape (network, [tf.shape(network)[0] * tf.shape(network)[1], tf.shape(network)[2], tf.shape(network)[3]])
#                network = tf.reshape (network, [-1, tf.shape(network)[2], tf.shape(network)[3]])
#                network = tf.reshape (network, [-1, 64, 3])
#                network = tf.reshape (network, [-1, 32, 3])
                network = tf.reshape (network, [-1, orig_shape[2], orig_shape[3]])
                logging.error('After reshape, TCN input is of shape: ' + str(network.get_shape()))
                network = tcn (network, [3, 3, 3, 3, 3, 3], 0, atten=False, dropout=tf.constant(0.0, dtype=tf.float32)) # TODO: The filters vector needs to be calculated, not hard set
                logging.error('After TCN, before reshape: ' + str(network.get_shape()))
#                network = tf.reshape (network, [-1, 41, 64, 3])
#                network = tf.reshape (network, [-1, 41, 32, 3])
                network = tf.reshape (network, [-1, orig_shape[1], orig_shape[2], orig_shape[3]])[:,:,1:,:] # <--- try [:,:,:-1,:] instead
#                network = tf.reshape (network, [-1, orig_shape[1], orig_shape[2], orig_shape[3]])[:,:,:-1,:] Nope
                logging.error('After TCN and reshape: ' + str(network.get_shape()))
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "MaxPooling":
                network = tflearn.layers.conv.max_pool_2d(network, layer["strides"])
            elif layer["type"] == "AveragePooling":
                network = tflearn.layers.conv.avg_pool_2d(network, layer["strides"])
            elif layer["type"] == "LocalResponseNormalization":
                network = tflearn.layers.normalization.local_response_normalization(network)
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, 1, [1, width], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network = network[:, :, 0, 0]
                btc_bias = tf.ones((self.input_num, 1))
                self.add_layer_to_dict(layer["type"], network)
                network = tf.concat([btc_bias, network], 1)
                network = tflearn.layers.core.activation(network, activation="softmax")
                self.add_layer_to_dict(layer["type"], network, weights=False)
            elif layer["type"] == "Output_WithW":
                network = tflearn.flatten(network)
                network = tf.concat([network,self.previous_w], axis=1)
                network = tflearn.fully_connected(network, self._rows+1,
                                                  activation="softmax",
                                                  regularizer=layer["regularizer"],
                                                  weight_decay=layer["weight_decay"])
            elif layer["type"] == "EIIE_Output_WithW":                                      # <---  what we use
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                network = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network = network[:, :, 0, 0]
                #btc_bias = tf.zeros((self.input_num, 1))
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
#                                       initializer=tf.zeros_initializer)
                                       initializer=tf.ones_initializer)
                # self.add_layer_to_dict(layer["type"], network, weights=False)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                self.add_layer_to_dict('voting', network, weights=False)
                network = tflearn.layers.core.activation(network, activation="softmax")
                self.add_layer_to_dict('softmax_layer', network, weights=False)

            elif layer["type"] == "EIIE_Output_WithWC": # Couldn't make this work. Not sure if have been useful at all.
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                logging.error('previous_w shape before reshaping: ' + str(self.previous_w.shape))
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                logging.error('previous_w shape after reshaping: ' + str(w.shape))
#                c = tf.reshape(self.cctv, [-1, int(height), 1, 1]) # Incompatible shapes: [11,42] vs. [109,42]
#                 c = tf.reshape(self.cctv, [-1, 41, 1, 1]) # Incompatible shapes: [11,42] vs. [109,42]
#                c = tf.reshape(self.cctv, [-1, 1, 1, 1]) # Dimension 1 in both shapes must be equal, but are 41 and 1. Shapes are [?,41,1] and [451,1,1]. for 'concat_1' (op: 'ConcatV2') with input shapes: [?,41,1,11], [451,1,1,1], [] and with computed input tensors: input[2] = <3>
                # Ah, previous_w is already a (?, 41) shaped. Where do we get this ? from? from a tf.placeholder. Fine, but where do we fill it with data? in nnagent.
#                c = tf.reshape(self.cctv, [-1, 41, 1, 1])
                # For concat, all dims must be equal except the axis specified (which'll be the sum from individual tensors)
                logging.error('consumptions shape before reshaping: ' + str(self.consumptions_vector.shape))
#                c = tf.reshape(self.consumptions_vector, [109, int(height), 1, 1])
#                c = tf.ones([109, 1, 1]) * self.consumptions_vector
                c = tf.reshape(self.consumptions_vector, [-1, int(height), 1, 1])
                logging.error('consumptions shape after reshaping: ' + str(c.shape))
                cw = tf.concat([c, w], axis=3)
                logging.error('cw shape after reshaping: ' + str(cw.shape))
                logging.error('network before concatenation: ' + str(network.shape))
                network = tf.concat([network, cw], axis=3)
#                network = tf.concat([network, c], axis=3)
#                network = tf.concat([network, w], axis=3)
                logging.error('network after concatenation: ' + str(network.shape))
                network = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network = network[:, :, 0, 0]
                #btc_bias = tf.zeros((self.input_num, 1))
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
#                                       initializer=tf.zeros_initializer)
                                       initializer=tf.ones_initializer)
                # self.add_layer_to_dict(layer["type"], network, weights=False)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                self.add_layer_to_dict('voting', network, weights=False)
                network = tflearn.layers.core.activation(network, activation="softmax")
                self.add_layer_to_dict('softmax_layer', network, weights=False)

            elif layer["type"] == "EIIE_LSTM" or\
                            layer["type"] == "EIIE_RNN":
                network = tf.transpose(network, [0, 2, 3, 1])
                resultlist = []
                reuse = False
                for i in range(self._rows):
                    if i > 0:
                        reuse = True
                    if layer["type"] == "EIIE_LSTM":
                        result = tflearn.layers.lstm(network[:, :, :, i],
                                                     int(layer["neuron_number"]),
                                                     dropout=layer["dropouts"],
                                                     scope="lstm"+str(layer_number),
                                                     reuse=reuse)
                    else:
                        result = tflearn.layers.simple_rnn(network[:, :, :, i],
                                                           int(layer["neuron_number"]),
                                                           dropout=layer["dropouts"],
                                                           scope="rnn"+str(layer_number),
                                                           reuse=reuse)
                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, self._rows, 1, int(layer["neuron_number"])])
            elif layer["type"] == "BatchNormalization":
                network = tf.layers.batch_normalization(network, axis=-1)
            elif layer["type"] == "ReLU":
                network = tflearn.activations.relu(network)
            elif layer["type"] == "EIIE_Output_WithW_WithBN":
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                network = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network = tf.layers.batch_normalization(network, axis=-1)
                network = network[:, :, 0, 0]
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
                                           initializer=tf.zeros_initializer)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                self.add_layer_to_dict('voting', network, weights=False)
                network = tflearn.layers.core.activation(network, activation="softmax")
                self.add_layer_to_dict('softmax_layer', network, weights=False)
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

