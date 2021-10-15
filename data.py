# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=missing-docstring
from __future__ import print_function
from tool_wsoh import *
import os
import numpy as np
# from sklearn.datasets import fetch_mldata


# def get_mnist():
#     """ Gets MNIST dataset """
#
#     np.random.seed(1234) # set seed for deterministic ordering
#     data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#     data_path = os.path.join(data_path, '../../data')
#     mnist = fetch_mldata('MNIST original', data_home=data_path)
#     p = np.random.permutation(mnist.data.shape[0])
#     X = mnist.data[p].astype(np.float32)*0.02
#     Y = mnist.target[p]
#     return X, Y

# 크롤링 리뷰 벡터로 불러오는 함수
def load_crawling_data():
    from tqdm import tqdm
    from gensim.models import Word2Vec

    np_size = 100
    window_size = 30
    min_count_size = 50
    sg_type = 1 # 1 = skip-gram, 0 = CBOW
    hs_type = 0 # default값 0, 1이면 softmax 함수 사용
    iter_cnt = 100 # epoch을 나누어  실행하는 횟수

    file_name = 'maindata_test_labeled_1_2ndtry_4'
    df = csv_reader(file_name)
    sentence_data = reviews_parcing(file_name)

    model = Word2Vec(sentences=sentence_data, size=np_size, window=window_size, min_count=min_count_size, workers=4, sg=sg_type)

    model.save(f"model_{file_name}")

    x = np.zeros((np_size,), dtype='float64') # 0으로 채운 ndarray(100,)
    y = np.array(df['total_score'], dtype='int64') # 점수 ndarray(5000,)로 만들기
    nwords = 0.
    counter = 0.
    raw_reivews = []
    token_reviews = []
    y_list = []
    print(df['review'][0])
    # GPU ver.
    idx_to_key = model.wv.index2word # 모델의 사전에 있는 단어명을 담은 리스트
    # print(idx_to_key)

    index2word_set = set(idx_to_key) # 속도를 위해 set 형태로 초기화
    # print(index2word_set)
    for idx in tqdm(range(len(sentence_data)), desc="벡터화"):
        featureVec = np.zeros((np_size,), dtype='float64')
        for word in sentence_data[idx]:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model.wv[word])
        is_all_zero = not np.any(featureVec)
        if is_all_zero:
            pass
        else:
            featureVec = np.divide(featureVec, nwords)
            x = np.append(x, featureVec, axis=0)
            # y의 개수를 x만큼 맞추기 위해

            y_list.append(y[idx])
            raw_reivews.append(df['review'][idx])
            # 토크나이징 리뷰 데이터도 추가해줘야 한다
            token_reviews.append(df['tokenized_review'][idx])

        counter += 1

    y = np.array(y_list, dtype='int64')
    x = x[np_size:]
    x = x.reshape(y.size, -1)
    print(len(y), 'train sequences')
    print('x_train shape:', x.shape)
    print('y shape:', y.shape)
    print("raw_reviews shape:", len(raw_reivews))
    print("token_reviews shape:", len(token_reviews))

    # print(x.astype(float))
    return x.astype(float), y, raw_reivews, token_reviews



