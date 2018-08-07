# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module provides wrappers for variants of RNN in Tensorflow
"""

import tensorflow as tf
import tensorflow.contrib as tc


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True, state=None,
        history=False):
    """
    实现 (Bi-)LSTM, (Bi-)GRU 和 (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
        state:
        history:
    Returns:
        RNN outputs and final state
    """
    if history:
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c, h = state
                state = h
        else:
            # 双向lstm,前向细胞、反向细胞
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32, initial_state_fw=state,
                initial_state_bw=state
             )
        # 获取双向状态
            state_fw, state_bw = state
            if rnn_type.endswith('lstm'):
                c_fw, h_fw = state_fw
                c_bw, h_bw = state_bw
                # 双向历史信息
                state_fw, state_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                state = tf.concat([state_fw, state_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                state = state_fw + state_bw
        return outputs, state
    else:
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c, h = state
                state = h
        else:
            # 双向lstm,前向细胞、反向细胞
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
             )
        # 获取双向状态
            state_fw, state_bw = state
            if concat:
                outputs = tf.concat(outputs, 2)
            else:
                outputs = outputs[0] + outputs[1]
        return outputs, state_fw


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    获取循环神经网络的细胞
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:nz
        An RNN Cell
    """
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    if layer_num > 1:
        cell = tc.rnn.MultiRNNCell([cell]*layer_num, state_is_tuple=True)
    return cell


