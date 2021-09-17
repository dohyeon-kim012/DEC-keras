# # import pandas as pd
import numpy as np
from konlpy.tag import Okt
okt = Okt()
from keras.preprocessing.text import Tokenizer
# # from gensim.models import Word2Vec
# # tokenized_data = []
# # from keras.utils.np_utils import to_categorical
# #
# # print('Loading data...')
# # df = pd.read_csv("./data/crawling_data/reviewdata_hotelsdotcom_preprocessed.csv", index_col=[0], nrows=200)
# # df.dropna(subset=['preprocessed_review', 'total_score'], inplace=True)
# # df = df.reset_index(drop=True)
# # data = df['preprocessed_review']
# # y = (df['total_score'] / 2)
# # # print(y[5])
# # print(y)
# # print(len(data), 'train sequences')
# #
# # num_classes = np.max(y)
# # print(num_classes, 'classes')
# #
# # print('Vectorizing sequence data...')
# # for review in data:
# #     temp = []
# #     oks = okt.pos(review, join=True)
# #     for ok in oks:
# #         text, tag = map(str, ok.split('/'))
# #         if tag not in ['Josa', 'Eomi', 'Punctation']:
# #             temp.append(text)
# #     tokenized_data.append(temp)
# #
# # model = Word2Vec(sentences=tokenized_data, window=5, min_count=10, workers=4, sg=1)
# # # x = np.zeros((len(tokenized_data), 100), dtype='float32')
# # x = np.empty((100,), dtype='float32')
# # # print(x)
# # # x_drop = np.delete(x, x, axis=0)
# # # print(x)
# # # print('x shape:', x.shape)
# # nwords = 0.
# # y_list = []
# # counter = 0.
# # idx_to_key = model.wv.index_to_key
# # key_to_idx = model.wv.key_to_index
# # index2word_set = set(idx_to_key)
# # for review in tokenized_data:
# #     featureVec = np.zeros((100,), dtype='float32')
# #     for word in review:
# #         if word in index2word_set:
# #             # print(word)
# #             nwords = nwords + 1.
# #             featureVec = np.add(featureVec, model.wv[word])
# #             # featureVec[np.arange(featureVec, model.wv[word])] = 1   --> 실패!
# #             # featureVec = to_categorical(model.wv[word], num_classes=(len(model.wv[word])))
# #     is_all_zero = not np.any(featureVec)
# #     if is_all_zero:
# #         pass
# #     else:
# #         # print('yes')
# #         featureVec = np.divide(featureVec, nwords)
# #         # print('featureVec :', int(counter), featureVec)
# #         # x[int(counter)] = featureVec
# #         x = np.append(x, featureVec, axis=0)
# #         # print('x :', x)
# #         y_list.append(y[int(counter)])
# #         # print(y[int(counter)])
# #         # print('y_list :', y_list)
# #     counter += 1
# # drop_data = x[:100]
# #
# # # print(x[:100])
# # y = np.array(y_list, dtype='int64')
# # x = x[100:]
# # x = x.reshape(y.size, -1)
# # #
# # #     # print('x :', counter, x[int(counter)])
# # # # x = to_categorical(x, num_classes=x)
# # # # x = np.delete(x, [1, ], axis=0)
# # # # x = x.reshape(830, 100)
# # print('x :', x)
# # print('x_train shape:', x.shape)
# # print('y :', y)
# # print('y_train shape:', y.shape)
# # # # print(model.wv.vectors.shape) ==> (242, 100)
# # # # word_vectors = model.wv
# # # # vocabs = word_vectors.key_to_index
# # # # word_vectors_list = [word_vectors[v] for v in vocabs]
# # # # x = np.array(word_vectors_list)
# # # # print(vocabs)
# # # # print(word_vectors_list[:10])
# # # # print(x[0])
# # #
# # # # # scale to [0,1]
# # # # from sklearn.preprocessing import MinMaxScaler
# # # # x_scaled = MinMaxScaler().fit_transform(x)
# # # # x = np.identity(3)[to_array.astype(int)]
# # # # print(to_array)
# # # # print(x_scaled)
# # #
# # # # from keras.preprocessing.text import Tokenizer
# # # # from keras.datasets import imdb
# # # # max_words = 1000
# # # #
# # # # print('Loading data...')
# # # # (x1, y1), (x2, y2) = imdb.load_data(num_words=max_words)
# # # # x = np.concatenate((x1, x2))
# # # # y = np.concatenate((y1, y2))
# # # # print(len(x), 'train sequences')
# # # #
# # # # num_classes = np.max(y) + 1
# # # # print(num_classes, 'classes')
# # # #
# # # # print('Vectorizing sequence data...')
# # # # tokenizer = Tokenizer(num_words=max_words)
# # # # x = tokenizer.sequences_to_matrix(x, mode='binary')
# # # # print('x_train shape:', x.shape)
# # # # print(type(x))
# # # # print(type(y))
# # # # return x.astype(float), y
# # #
# # #
# #
# # # w = np.zeros((6, 6), dtype=np.int64)
# # # w[4, 4] += 1
# # # print(w)
# #
# # # import numpy as np
# #
# # a = np.zeros((100,), float)
# # # if np.all(a == 0) is True:
# # #     print('000')
# # # print(np.all(a == 0))
# # # is_all_zero = not np.any(a)
# # # if is_all_zero:
# # #     print('yes')
#
# import pandas as pd
# from konlpy.tag import Okt
# okt = Okt()
# from gensim.models import Word2Vec
#
# tokenized_data = []
# y_list = []
#
# print('Loading data...')
# df = pd.read_csv("./data/crawling_data/reviewdata_hotelsdotcom_preprocessed.csv", index_col=[0], nrows=5000)
# df.dropna(subset=['preprocessed_review', 'total_score'], inplace=True)
# df = df.reset_index(drop=True)
# data = df['review']
# y = (df['total_score'] / 2)
#
# # num_classes = np.max(y)
# # print(num_classes, 'classes')
# '''
#  {'Adjective': '형용사', 'Adverb': '부사', 'Alpha': '알파벳', 'Conjunction': '접속사', 'Determiner': '관형사',
#     'Eomi': '어미', 'Exclamation': '감탄사', 'Foreign': '외국어, 한자 및 기타기호', 'Hashtag': '트위터 해쉬태그',
#     'Josa': '조사', 'KoreanParticle': '(ex: ㅋㅋ)', 'Noun': '명사', 'Number': '숫자', 'PreEomi': '선어말어미',
#     'Punctuation': '구두점', 'ScreenName': '트위터 아이디', 'Suffix': '접미사', 'Unknown': '미등록어', 'Verb': '동사'}
#     '''
# print('Vectorizing sequence data...')
# for review in data:
#     temp = []
#     oks = okt.pos(review, norm=True) #, join=True
#     for ok in oks:
#         # text, tag = map(str, ok.split('/'))
#         # # if tag not in ['Josa', 'Eomi', 'Punctation']:
#         # if tag in ['Noun']:
#             temp.append(ok)
#     tokenized_data.append(temp)
# print(tokenized_data)

#########################################################################################
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
okt = Okt()
from gensim.models import Word2Vec

tokenized_data = []
y_list = []

print('Loading data...')
df = pd.read_csv("./data/crawling_data/reviewdata_hotelsdotcom_preprocessed.csv", index_col=[0], nrows=50000)
df.dropna(subset=['preprocessed_review', 'total_score'], inplace=True)
df = df.reset_index(drop=True)
data = df['preprocessed_review']
y = (df['total_score'] / 2)

num_classes = np.max(y)
print(num_classes, 'classes')

print('Vectorizing sequence data...')
for review in data:
    temp = []
    oks = okt.pos(review, join=True)
    for ok in oks:
        text, tag = map(str, ok.split('/'))
        if tag not in ['Josa', 'Eomi', 'Punctation']:
            temp.append(text)
    tokenized_data.append(temp)
tokenizer.fit_on_texts(tokenized_data)
# print(tokenizer.word_index)

# 빈도수 검사
threshold = 2
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if value < threshold:
        rare_cnt += 1
        rare_freq = rare_freq + value

# print("단어 집합의 크기 : {}".format(total_cnt))
# print("등장 빈도가 {}번 이하인 희귀 단어의 수 : {}".format(threshold-1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율 : {:.2f}%".format((rare_cnt / total_cnt) * 100))
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율 : {:.2f}%".format((rare_freq / total_freq) * 100))

vocab_size = total_cnt - rare_cnt + 1
# print(vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(tokenized_data)

x_encoded = tokenizer.texts_to_matrix(tokenized_data, mode='freq')
print(x_encoded[:10])
print(x_encoded.shape)
# drop_train = [index for index, sentence in enumerate(x_encoded) if len(sentence) < 1]
# print(drop_train) -> 없음
# x_encoded_final = np.delete(x_encoded, drop_train, axis=0)


############################################################



