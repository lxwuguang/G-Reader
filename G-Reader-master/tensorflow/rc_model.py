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
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import math
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import very_compute_bleu_rouge
from utils import rabel_compute_bleu_rouge
from utils import rouge_compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
from layers.match_layer import SelfMatchingLayer
from collections import defaultdict


def get_score(result_dir):
    pred_answers = []
    with open(result_dir) as fin:
        for lidx_a, line_a in enumerate(fin):
            em_answer = json.loads(line_a.strip())
            pred_answers.append(em_answer)

    # 如果有参考答案，计算bleu和rouge分数
    pred_dict, ref_dict = defaultdict(dict), defaultdict(dict)
    num = 0
    for answer in pred_answers:
        question_id = answer['question_id']
        if len(answer['ref_answers']) > 0:
            num += 1
            pred_dict[question_id][str(num)] = normalize(answer['answers'])
            ref_dict[question_id][str(num)] = normalize(answer['ref_answers'])

    rouge, bleu = very_compute_bleu_rouge(pred_dict, ref_dict)

    return rouge, bleu


def label(result_dir, model_dir, result_prefix):
    pred_answers = []
    with open(result_dir) as fin:
        for lidx_a, line_a in enumerate(fin):
            em_answer = json.loads(line_a.strip())
            pred_answers.append(em_answer)

    # 如果有参考答案，计算bleu和rouge分数
    pred_dict, ref_dict = defaultdict(dict), defaultdict(dict)
    for answer in pred_answers:
        question_id = answer['question_id']
        if len(answer['ref_answers']) > 0:
            pred_dict[question_id][answer['ans_id']] = normalize(answer['answers'])
            ref_dict[question_id][answer['ans_id']] = normalize(answer['ref_answers'])

    result = rabel_compute_bleu_rouge(pred_dict, ref_dict)

    if model_dir is not None and result_prefix is not None:
        result_file = os.path.join(model_dir, result_prefix + '.json')
        with open(result_file, 'w') as fout:
            for ans in result:
                fout.write(json.dumps(ans, encoding='utf8', ensure_ascii=False) + '\n')


def rouge(result_dir, model_dir, result_prefix):
    pred_answers = []
    with open(result_dir) as fin:
        for lidx_a, line_a in enumerate(fin):
            em_answer = json.loads(line_a.strip())
            pred_answers.append(em_answer)

    pred_dict, ref_dict = defaultdict(dict), defaultdict(dict)
    for answer in pred_answers:
        question_id = answer['question_id']
        pred_dict[question_id][answer['ans_id']] = normalize(answer['answers'])

    for key in pred_dict.keys():
        for ans in pred_dict[key].keys():
            ref_answers = []
            for ref_ans in pred_dict[key].keys():
                if ans == ref_ans:
                    continue
                ref_answers.append(pred_dict[key][ref_ans])
            ref_dict[key][ans] = normalize(ref_answers)

    result = rouge_compute_bleu_rouge(pred_dict, ref_dict)

    if model_dir is not None and result_prefix is not None:
        result_file = os.path.join(model_dir, result_prefix + '.json')
        with open(result_file, 'w') as fout:
            for ans in result:
                fout.write(json.dumps(ans, encoding='utf8', ensure_ascii=False) + '\n')


class RCModel(object):
    """
    实现阅读理解模型
    """
    def __init__(self, vocab, args):

        # 日志
        self.logger = logging.getLogger("brc")

        # 基础设置
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # 长度限制
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # 词汇表
        self.vocab = vocab

        # 会话信息
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # 存储信息
        self.saver = tf.train.Saver()

        # 模型初始化
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        建立tf的运算图
        1.初始化占位符 2.词嵌入 3.编码 4.匹配 5.融合 6.解码 7.计算损失 8.创建训练节点
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('建立运算图用时为: {}  秒'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('模型中一共有 {} 个变量'.format(param_num))

    def _setup_placeholders(self):
        """
        占位符
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])

        self.context2question_attn = tf.placeholder(tf.float32, [None, None, None])
        self.question2context_attn = tf.placeholder(tf.float32, [None, None, None])
        self.selfmatch_mat = tf.placeholder(tf.float32, [None, None, None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        嵌入层.问题和文章共享同样的嵌入方式
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def _encode(self):
        """
        使用两个双向LSTM分别对问题和文章编码
        问题作为历史信息要流入到文章中
        """

        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, question_state = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size, state=question_state
                                        , history=True)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        阅读理解模型的核心.使用BiDAF或MLSTM来获取文章对问题的感知情况
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)

        self.match_p_encodes, self.context2question_attn, self.question2context_attn, _ = match_layer.match(
            self.sep_p_encodes, self.sep_q_encodes, self.p_length, self.q_length)

        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        match之后，使用Bi-LSTM来融合上下文信息
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

        with tf.variable_scope('self-matching'):
            match_layer = SelfMatchingLayer(self.hidden_size)
            tem_encodes = tf.identity(self.fuse_p_encodes)
            self_matching_encodes, _ = match_layer.match(self.fuse_p_encodes, tem_encodes, self.p_length)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self_matching_encodes, self.dropout_keep_prob)
            else:
                self.fuse_p_encodes = self_matching_encodes
            self.selfmatch_mat = self.fuse_p_encodes

    def _decode(self):
        """
        使用Pointer-Net来获取每一个位置作为答案起点的概率和终点的概率

        注意到：
        把多个段落fuse_p_encodes连接起来
        由于同一文档中问题的编码是相同的，所以选择第一个即可
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]

            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )

            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):
        """
        损失函数
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            负对数似然损失
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses
        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        选择训练算法并以此创建一个训练操作
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('不支持的优化器: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        单次训练模型
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('第 {} 到第 {}批训练集的平均损失为 {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0):
        """
        训练模型
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        for epoch in range(1, epochs + 1):
            self.logger.info('第{}次训练模型 '.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('该批次的平均训练损失 {} is {}'.format(epoch, train_loss))

            self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None):
        """
        评估模型在验证集上的表现，如果指定了保存，则将把结果保存
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        count = 0
        for b_itx, batch in enumerate(eval_batches):
            count += 1
            if count % 100 == 0:
                self.logger.info(count)
            if batch['passage_length'][0] <= 0:
                continue
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}

            start_probs, end_probs, loss, context2question_attn, question2context_attn, selfmatch_mat = self.sess.run([
                self.start_probs, self.end_probs, self.loss, self.context2question_attn,
                self.question2context_attn, self.selfmatch_mat], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):
                best_answer, seg_answers, best_score, match_mat = self.find_best_answer(
                    sample, start_prob, end_prob, padded_p_len, context2question_attn, question2context_attn,
                    selfmatch_mat)
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type': sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'question_tokens': sample['question'],
                                         'ref_answers': sample['answers'],
                                         'best_score': str(best_score),
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': [],
                                         'match_mat': match_mat,
                                         'seg_answers': [seg_answers]})
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'question_tokens': sample['question'],
                                         'answers': [best_answer],
                                         'best_score': str(best_score),
                                         'entity_answers': [[]],
                                         'yesno_answers': [],
                                         'match_mat': match_mat,
                                         'seg_answers': [seg_answers]})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, encoding='utf8', ensure_ascii=False) + '\n')

            self.logger.info('保存 {} 结果到 {}'.format(result_prefix, result_file))

        # 该平均损失对于测试集是无效的，因为测试集没有标注答案
        ave_loss = 1.0 * total_loss / total_num
        # 如果有参考答案，计算bleu和rouge分数
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len, context2question_attn, question2context_attn
                         , selfmatch_mat):
        """
        根据起始概率于每个段落找到最好的答案
        """
        best_p_idx, best_span, best_score = None, None, 0
        match_mat = []
        total_score = 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            total_score += score
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
            seg_answers = ['']
        else:
            c2q_mat = context2question_attn[best_p_idx][best_span[0]: best_span[1] + 1]
            match_mat.append(str(np.sum(c2q_mat)))
            match_mat.append(str(np.mean(c2q_mat)))
            match_mat.append(str(np.max(c2q_mat)))
            match_mat.append(str(np.min(c2q_mat)))

            q2c_mat = question2context_attn[best_p_idx][best_span[0]: best_span[1] + 1]
            match_mat.append(str(np.sum(q2c_mat)))
            match_mat.append(str(np.mean(q2c_mat)))
            match_mat.append(str(np.max(q2c_mat)))
            match_mat.append(str(np.min(q2c_mat)))

            se_mat = selfmatch_mat[best_p_idx][best_span[0]: best_span[1] + 1]
            match_mat.append(str(np.sum(se_mat)))
            match_mat.append(str(np.mean(se_mat)))
            match_mat.append(str(np.max(se_mat)))
            match_mat.append(str(np.min(se_mat)))

            start_span = best_p_idx * padded_p_len+best_span[0]
            end_span = best_p_idx * padded_p_len+best_span[1] + 1
            start_prob_mat = start_prob[start_span: end_span]
            match_mat.append(str(start_prob_mat[0]))
            match_mat.append(str(np.sum(start_prob_mat)))
            match_mat.append(str(np.mean(start_prob_mat)))
            match_mat.append(str(np.max(start_prob_mat)))
            match_mat.append(str(np.min(start_prob_mat)))

            end_prob_mat = end_prob[start_span: end_span]
            match_mat.append(str(end_prob_mat[-1]))
            match_mat.append(str(np.sum(end_prob_mat)))
            match_mat.append(str(np.mean(end_prob_mat)))
            match_mat.append(str(np.max(end_prob_mat)))
            match_mat.append(str(np.min(end_prob_mat)))

            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
            seg_answers = sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1]
        best_score = best_score/total_score
        return best_answer, seg_answers, best_score, match_mat

    """
                match_mat = []
                c2q_sum_mat = np.array(context2question_attn[best_p_idx][best_span[0]: best_span[1] + 1]).sum(axis=0)
                q2c_sum_mat = np.array(question2context_attn[best_p_idx][best_span[0]: best_span[1] + 1]).sum(axis=0)
                c2q_mean_mat = np.array(context2question_attn[best_p_idx][best_span[0]: best_span[1] + 1]).mean(axis=0)
                q2c_mean_mat = np.array(question2context_attn[best_p_idx][best_span[0]: best_span[1] + 1]).mean(axis=0)
                c2q_max_mat = np.array(context2question_attn[best_p_idx][best_span[0]: best_span[1] + 1]).max(axis=0)
                q2c_max_mat = np.array(question2context_attn[best_p_idx][best_span[0]: best_span[1] + 1]).max(axis=0)
                c2q_min_mat = np.array(context2question_attn[best_p_idx][best_span[0]: best_span[1] + 1]).min(axis=0)
                q2c_min_mat = np.array(question2context_attn[best_p_idx][best_span[0]: best_span[1] + 1]).min(axis=0)
                the_mat = np.concatenate((np.add(c2q_sum_mat, q2c_sum_mat), np.add(c2q_mean_mat, q2c_mean_mat),
                                          np.add(c2q_max_mat, q2c_max_mat), np.add(c2q_min_mat, q2c_min_mat)), axis=0)
                for index in range(len(the_mat)):
                    match_mat.append(str(the_mat[index]))
    """

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        根据每个段落起点乘以终点的最大值来找最好的答案
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        保存模型
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('保存模型到 {}, 其前缀为 {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        重载模型
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('重载 {} 模型, 其前缀为 {}'.format(model_dir, model_prefix))
