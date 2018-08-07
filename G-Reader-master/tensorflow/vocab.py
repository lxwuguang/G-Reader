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
This module implements the Vocab class for converting string to id and back
"""

import numpy as np


class Vocab(object):
    """
    通过tokens一致的嵌入，实现一个词汇表来存储数据里的tokens（词语）
    """
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        获取词汇表的大小
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def load_from_file(self, file_path):
        """
        从文件路径加载词汇表
        Args:
            file_path: a file with a word in each line
        """
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, token):
        """
        获得某个词语的id，如果词语不在词汇表中，则返回未知词标识
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        获取与id相对应的词语，如果id不在词汇表中，返回未知标识
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, token, cnt=1):
        """
        把词语加入到词汇表
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        过滤掉一些低频词
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        随机初始化词向量
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def load_pretrained_embeddings(self, embedding_path):
        """
        根据文件路径，加载预训练的词向量
        不在该词向量集中的词语将被过滤
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0].decode('utf8')
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # 重构词语和id的映射关系
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # 加载词嵌入
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def convert_to_ids(self, tokens):
        """
        将一组词语转化为id序列
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        将一组id序列转化为一组词语
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
