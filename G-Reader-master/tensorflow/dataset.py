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
This module implements data process strategies.
"""
import os
import json
import logging
import numpy as np
from collections import Counter


class BRCDataset(object):
    """
    该模块实现加载和使用百度阅读理解数据集的api
    """

    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('训练集规模: {} 个问题.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('验证集规模: {} 个问题.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('测试集规模: {} 个问题.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        加载数据集
        """
        with open(data_path) as fin:
            data_set = []
            filter_long_para = 0
            filter_answer_spans = 0
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                sample['question_tokens'] = sample['segmented_question']
                sample['passages'] = []
                # 记录正确答案所在的段落
                if train:
                    if len(sample['answer_spans']) == 0:
                        filter_answer_spans += 1
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        filter_answer_spans += 1
                        continue
                    if sample['answer_docs'][0] >= len(sample['documents']):
                        self.logger.info('like that:')
                        self.logger.info(len(sample['documents']))
                        self.logger.info(sample['answer_docs'][0])
                        continue
                    if len(sample['documents'][sample['answer_docs'][0]]['segmented_paragraphs']) > self.max_p_num:
                        filter_long_para += 1
                        continue

                    # 找到被选中的文章
                    docs_index = sample['answer_docs'][0]
                    # 记录被选中的段落
                    sample['answer_passages'] = [sample['documents'][docs_index]['most_related_para']]
                    # 把文章的每一段落添加到训练集
                    for para in sample['documents'][docs_index]['segmented_paragraphs']:
                        sample['passages'].append({'passage_tokens': para})
                else:
                    if len(sample['documents']) == 0:
                        self.logger.info(sample)
                        continue
                    for para in sample['documents'][0]['segmented_paragraphs']:
                        sample['passages'].append({'passage_tokens': para})
                    if len(sample['passages']) == 0:
                        self.logger.info('empty passages.')
                        continue
                    if len(sample['segmented_question']) == 0:
                        self.logger.info('empty question.')
                        self.logger.info(sample)
                        continue
                data_set.append(sample)

            if train:
                self.logger.info('训练集中，有{}个问题因段落过长而被过滤'.format(filter_long_para))
                self.logger.info('训练集中，有{}个问题因answer_span问题而被过滤'.format(filter_answer_spans))

        return data_set

    def _one_mini_batch(self, data, indices, pad_id,train=False):
        """
        一个最小的批次
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        # 最大段落数
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if train:
                    if pidx < len(sample['passages']):
                        batch_data['question_token_ids'].append(sample['question_token_ids'])
                        batch_data['question_length'].append(len(sample['question_token_ids']))
                        passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                        batch_data['passage_token_ids'].append(passage_token_ids)
                        batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))

                    else:
                        batch_data['question_token_ids'].append([])
                        batch_data['question_length'].append(0)
                        batch_data['passage_token_ids'].append([])
                        batch_data['passage_length'].append(0)

                else:
                    if pidx < len(sample['passages']) and len(sample['passages'][pidx]['passage_token_ids']) > 0:
                        batch_data['question_token_ids'].append(sample['question_token_ids'])
                        batch_data['question_length'].append(len(sample['question_token_ids']))
                        batch_data['passage_token_ids'].append(passage_token_ids)
                        batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))

                    else:
                        batch_data['question_token_ids'].append([])
                        batch_data['question_length'].append(0)
                        batch_data['passage_token_ids'].append([])
                        batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # 一些样本的假答案，只用于测试
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        根据pad_id动态填充batch_data
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        遍历数据集里的所有词语
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        把原始数据集里的问题和文章转化为id序列
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        对于任一个指定的数据集(train/dev/test)都通用的batch
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('没有叫做" {} "的数据集'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]

            if set_name == 'train':
                yield self._one_mini_batch(data, batch_indices, pad_id, train=True)
            else:
                yield self._one_mini_batch(data, batch_indices, pad_id)
