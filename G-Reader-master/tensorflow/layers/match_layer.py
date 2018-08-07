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
该模块实现Match-LSTM和BiDAF算法
"""

import tensorflow as tf
import tensorflow.contrib as tc


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    实现M-LSTM注意力Cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    实现Match-LSTM层
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        根据问题编码来匹配文章编码
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    实现注意力流层来计算文本-问题、问题-文本的注意力
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        根据问题编码来匹配文章编码
        """
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                            [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, passage_encodes * context2question_attn, passage_encodes * question2context_attn, None


class SelfMatchingLayer(object):
    """
    Implements the self-matching layer.
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def getSelfMatchingCell(self, hidden_size, in_keep_prob=1.0):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def match(self, passage_encodes, whole_passage_encodes, p_length):
        with tf.variable_scope('self-matching'):
            # 创建cell
            # whole_passage_encodes 作为整体匹配信息

            # cell_fw = SelfMatchingCell(self.hidden_size, question_encodes)
            # cell_bw = SelfMatchingCell(self.hidden_size, question_encodes)

            cell_fw = self.getSelfMatchingCell(self.hidden_size)
            cell_bw = self.getSelfMatchingCell(self.hidden_size)

            # function:

            # self.context_to_attend = whole_passage_encodes
            # fc_context = W * context_to_attend
            self.fc_context = tc.layers.fully_connected(whole_passage_encodes, num_outputs=self.hidden_size,
                                                        activation_fn=None)
            ref_vector = passage_encodes
            # 求St的tanh部分
            G = tf.tanh(self.fc_context + tf.expand_dims(
                tc.layers.fully_connected(ref_vector, num_outputs=self.hidden_size, activation_fn=None), 1))
            # tanh部分乘以bias
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            # 求a
            scores = tf.nn.softmax(logits, 1)
            # 求c
            attended_context = tf.reduce_sum(whole_passage_encodes * scores, axis=1)
            # birnn inputs
            input_encodes = tf.concat([ref_vector, attended_context], -1)
            """
            gated
            g_t = tf.sigmoid( tc.layers.fully_connected(whole_passage_encodes,num_outputs=self.hidden_size,activation_fn=None) )
            v_tP_c_t_star = tf.squeeze(tf.multiply(input_encodes , g_t))
            input_encodes = v_tP_c_t_star
            """

            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=input_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)

            match_outputs = tf.concat(outputs, 2)
            match_state = tf.concat([state, state], 1)

            # state_fw, state_bw = state
            # c_fw, h_fw = state_fw
            # c_bw, h_bw = state_bw
        return match_outputs, match_state