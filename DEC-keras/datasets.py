import numpy as np
from tqdm import tqdm


def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model

    # im_h = x.shape[1]
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    # if flatten:
    #     add_layer = Flatten()
    # else:
    #     add_layer = GlobalMaxPool2D()
    # feature_model = Model(model.input, add_layer(model.output))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = preprocess_input(x)  # data - 127. #data/255.#
    features = feature_model.predict(x)
    print('Features shape = ', features.shape)

    return features


def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_pendigits(data_path='./data/pendigits'):
    import os
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tra -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.tes -P %s' % data_path)
        os.system('wget http://mlearn.ics.uci.edu/databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]
    print('data_train shape=', data_train.shape)

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]
    print('data_test shape=', data_test.shape)

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    print('pendigits samples:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y


def load_retures_keras():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import reuters
    max_words = 1000

    print('Loading data...')
    (x, y), (_, _) = reuters.load_data(num_words=max_words, test_split=0.)
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_imdb():
    from keras.preprocessing.text import Tokenizer
    from keras.datasets import imdb
    max_words = 1000

    print('Loading data...')
    (x1, y1), (x2, y2) = imdb.load_data(num_words=max_words)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    print(len(x), 'train sequences')

    num_classes = np.max(y) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x = tokenizer.sequences_to_matrix(x, mode='binary')
    print('x_train shape:', x.shape)

    return x.astype(float), y


def load_newsgroups():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    x_sparse = vectorizer.fit_transform(newsgroups.data)
    x = np.asarray(x_sparse.todense())
    y = newsgroups.target
    print('News group data shape ', x.shape)
    print("News group number of clusters: ", np.unique(y).size)
    return x, y


def load_cifar10(data_path='./data/cifar10'):
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    # if features are ready, return them
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy'):
        return np.load(data_path + '/cifar10_features.npy'), y

    # extract features
    features = np.zeros((60000, 4096))
    for i in range(6):
        idx = range(i*10000, (i+1)*10000)
        print("The %dth 10000 samples" % i)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/cifar10_features.npy', features)
    print('features saved to ' + data_path + '/cifar10_features.npy')

    return features, y


def load_stl(data_path='./data/stl'):
    import os
    assert os.path.exists(data_path + '/stl_features.npy') or not os.path.exists(data_path + '/train_X.bin'), \
        "No data! Use %s/get_data.sh to get data ready, then come back" % data_path

    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))

    # if features are ready, return them
    if os.path.exists(data_path + '/stl_features.npy'):
        return np.load(data_path + '/stl_features.npy'), y

    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)

    # extract features
    features = extract_vgg16_features(x)

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    np.save(data_path + '/stl_features.npy', features)
    print('features saved to ' + data_path + '/stl_features.npy')

    return features, y


def load_crawling_data():
    import pandas as pd
    from konlpy.tag import Okt
    okt = Okt()
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    from transformers import BertTokenizer
    from gensim.models import Word2Vec
    tokenized_data = []
    y_list = []
    import gzip
    import pickle

    '''??????????????? ????????? ???????????? ??????
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
    '''
    # ???????????? ?????? ????????? ????????? ?????? ?????? ??????

    print('Loading data...')
    # df1 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_1.csv", index_col=[0])
    # df2 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_2.csv", index_col=[0])
    # df3 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_3.csv", index_col=[0])
    # df4 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_4.csv", index_col=[0])
    # df5 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_5.csv", index_col=[0])
    # df = pd.concat([df1[:30000], df2[:30000], df3[:30000], df4[:30000], df5[:30000]])
    # df = df.loc[:, ['total_score', 'tokenized_review']]
    # df_dropped = df.dropna(axis=0)
    # df_shuffled = df_dropped.sample(frac=1).reset_index(drop=True)
    # print(df_shuffled)

    df1 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_1.csv", index_col=[0])
    # df2 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_2.csv", index_col=[0])
    # df3 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_3.csv", index_col=[0])
    # df4 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_4.csv", index_col=[0])
    df5 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_5.csv", index_col=[0])
    # ???????????? ???????????? ?????? df??????, ????????? ????????? ??????
    df = pd.concat([df1[:30000], df5[:30000]])  # , df2[:30000], df3[:30000], df4[:30000],
    df = df.loc[:, ['total_score', 'tokenized_review']]
    df_dropped = df.dropna(axis=0)
    data = df_dropped.sample(frac=1).reset_index(drop=True)
    df_shuffled = df_dropped.sample(frac=1).reset_index(drop=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(s) for s in data['preprocessed_review']]
    # ???????????? ???????????? ?????? df??????, ????????? ????????? ??????

    # x_list = df_shuffled['tokenized_review'].values.tolist()
    # data = []
    # for i in tqdm(range(len(x_list)), desc='okt ?????? data >>> pkl list data'):
    #     x_list[i] = x_list[i].replace("[", "")
    #     x_list[i] = x_list[i].replace("]", "")
    #     x_list[i] = x_list[i].replace(",", "")
    #     x_list[i] = x_list[i].replace("'", "")
    #     x_list[i] = x_list[i].split()
    #
    #     data.append(x_list[i])
    # print(data[149921])
    # df = pd.read_csv("./reviewdata_hotelsdotcom_okt.csv", index_col=[0], nrows=300)
    # df.dropna(subset=['tokenized_review', 'total_score'], inplace=True)
    # df = df_shuffled.reset_index(drop=True)
    # tokenized_data = df['tokenized_review']
    # y = (df['total_score'] / 2)
    y = np.array(df_shuffled['total_score'], dtype='int64')
    raw_data = []
    # for num in range(1, 5):
    # with gzip.open('./data/crawling_data/pkl_list_review_data_by_total_score_4.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     data = data[:y.size]

    num_classes = np.max(y)
    print(num_classes, 'classes')

    # tokenizer.fit_on_texts(tokenized_data)
    # threshold = 3
    # total_cnt = len(tokenizer.word_index)
    # rare_cnt = 0
    # total_freq = 0
    # rare_freq = 0
    #
    # for key, value in tokenizer.word_counts.items():
    #     total_freq = total_freq + value
    #
    #     if value < threshold:
    #         rare_cnt += 1
    #         rare_freq = rare_freq + value

    # print("?????? ????????? ?????? : {}".format(total_cnt))
    # print("?????? ????????? {}??? ????????? ?????? ????????? ??? : {}".format(threshold-1, rare_cnt))
    # print("?????? ???????????? ?????? ????????? ?????? : {:.2f}%".format((rare_cnt / total_cnt) * 100))
    # print("?????? ?????? ???????????? ?????? ?????? ?????? ?????? ?????? : {:.2f}%".format((rare_freq / total_freq) * 100))

    # vocab_size = total_cnt - rare_cnt + 1
    # print(vocab_size)
    #
    # tokenizer = Tokenizer(vocab_size, oov_token='<oov>')
    # tokenizer.fit_on_texts(tokenized_data)
    # x = tokenizer.texts_to_matrix(tokenized_data, mode='tfidf')


    # Word2Vec ??????
    min_count = 5
    model = Word2Vec(sentences=tokenized_texts, window=5, min_count=min_count, workers=4, sg=1)
    print(f'Vectorizing sequence data with min_count {min_count}...')
    # model = Word2Vec.load('model_mincnt_20')
    # x = np.zeros((len(tokenized_data), 100), dtype='float32')
    x = np.empty((100,), dtype='float32')
    nwords = 0.
    # counter = 0.
    # GPU ver.
    idx_to_key = model.wv.index2word
    # local ver.
    # idx_to_key = model.wv.index_to_key
    # key_to_idx = model.wv.key_to_index
    index2word_set = set(idx_to_key)
    for idx in tqdm(range(len(data)), desc="Word2Vec"):
        featureVec = np.zeros((100,), dtype='float32')
        for word in data[idx]:
            if word in index2word_set:
                nwords = nwords + 1.
                # print('len :', len(model.wv[word])) ---> 100
                featureVec = np.add(featureVec, model.wv[word])
                # featureVec[np.arange(featureVec, model.wv[word])] = 1   --> ??????!
                # featureVec = to_categorical(model.wv[word], num_classes=(len(model.wv[word])))
        # x?????? ??? ????????? ?????? y??? raw_data??? ??????
        is_all_zero = not np.any(featureVec)
        if is_all_zero:
            pass
        else:
            featureVec = np.divide(featureVec, nwords)
            # print('featureVec :', featureVec)
            x = np.append(x, featureVec, axis=0)
            # y??? ????????? x?????? ????????? ??????
            # y_list.append(y[int(counter)])
            y_list.append(y[idx])
            raw_data.append(df_shuffled['tokenized_review'][idx])
        # counter += 1
            # print('x :', counter, x[int(counter)])
    # x = to_categorical(x, num_classes=x)
    y = np.array(y_list, dtype='int64')
    x = x[100:]
    x = x.reshape(y.size, -1)
    print(len(y), 'train sequences')
    print('x_train shape:', x.shape)
    print('raw_review_data shape:', len(raw_data))
    print('y shape:', y.shape)

    return x.astype(float), y, raw_data


def load_crawling_data_fasttext():
    from konlpy.tag import Okt
    okt = Okt()
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()

    from gensim.models import Word2Vec, FastText
    tokenized_data = []
    y_list = []
    import pandas as pd
    import gzip
    import pickle

    '''??????????????? ????????? ???????????? ??????
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
    '''
    # ???????????? ?????? ????????? ????????? ?????? ?????? ??????

    print('Loading data...')
    df1 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_1.csv", index_col=[0])
    # df2 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_2.csv", index_col=[0])
    # df3 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_3.csv", index_col=[0])
    # df4 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_4.csv", index_col=[0])
    df5 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_5.csv", index_col=[0])
    # ???????????? ???????????? ?????? df??????, ????????? ????????? ??????
    df = pd.concat([df1[:30000], df5[:30000]])  # , df2[:30000], df3[:30000], df4[:30000],
    df = df.loc[:, ['total_score', 'tokenized_review']]
    df_dropped = df.dropna(axis=0)
    df_shuffled = df_dropped.sample(frac=1).reset_index(drop=True)
    data = df_shuffled['tokenized_review']
    # print(df_shuffled)
    # x_list = df_shuffled['tokenized_review'].values.tolist()
    # data = []
    # for i in tqdm(range(len(x_list)), desc='okt ?????? data >>> pkl list data'):
    #     x_list[i] = x_list[i].replace("[", "")
    #     x_list[i] = x_list[i].replace("]", "")
    #     x_list[i] = x_list[i].replace(",", "")
    #     x_list[i] = x_list[i].replace("'", "")
    #     x_list[i] = x_list[i].split()
    #
    #     data.append(x_list[i])
    # print(data[149921])
    # df = pd.read_csv("./reviewdata_hotelsdotcom_okt.csv", index_col=[0], nrows=300)
    # df.dropna(subset=['tokenized_review', 'total_score'], inplace=True)
    # df = df_shuffled.reset_index(drop=True)
    # tokenized_data = df['tokenized_review']
    # y = (df['total_score'] / 2)
    y = np.array(df_shuffled['total_score'], dtype='int64')
    raw_data = []
    # for num in range(1, 5):
    # with gzip.open('./data/crawling_data/pkl_list_review_data_by_total_score_4.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     data = data[:y.size]

    num_classes = np.unique(y)
    print(num_classes, 'classes')

    min_count = 10
    model = FastText(sentences=data, window=5, min_count=min_count, workers=4, sg=1)
    # model = FastText.load_fasttext_format('./cc.ko.300.bin.gz')
    print(f'Vectorizing sequence data with min_count {min_count}...')
    # model = Word2Vec.load('model_mincnt_20')
    # x = np.zeros((len(tokenized_data), 100), dtype='float32')
    x = np.empty((300,), dtype='float32')
    nwords = 0.
    # counter = 0.
    # GPU ver.
    idx_to_key = model.wv.index2word
    # local ver.
    # idx_to_key = model.wv.index_to_key
    # key_to_idx = model.wv.key_to_index
    index2word_set = set(idx_to_key)
    for idx in tqdm(range(len(data)), desc="Word2Vec"):
        featureVec = np.zeros((300,), dtype='float32')
        for word in data[idx]:
            if word in index2word_set:
                nwords = nwords + 1.
                # print('len :', len(model.wv[word])) ---> 100
                featureVec = np.add(featureVec, model.wv[word])
                # featureVec[np.arange(featureVec, model.wv[word])] = 1   --> ??????!
                # featureVec = to_categorical(model.wv[word], num_classes=(len(model.wv[word])))
        # x?????? ??? ????????? ?????? y??? raw_data??? ??????
        is_all_zero = not np.any(featureVec)
        if is_all_zero:
            pass
        else:
            featureVec = np.divide(featureVec, nwords)
            # print('featureVec :', featureVec)
            x = np.append(x, featureVec, axis=0)
            # y??? ????????? x?????? ????????? ??????
            # y_list.append(y[int(counter)])
            y_list.append(y[idx])
            review = []
            review.append(df_shuffled['review'][idx])
            review.append(df_shuffled['tokenized_review'][idx])
            raw_data.append(review)
        # counter += 1
            # print('x :', counter, x[int(counter)])
    # x = to_categorical(x, num_classes=x)
    y = np.array(y_list, dtype='int64')
    x = x[300:]
    x = x.reshape(y.size, -1)
    print(len(y), 'train sequences')
    print('x_train shape:', x.shape)
    print('raw_review_data shape:', len(raw_data))
    print('y shape:', y.shape)

    return x.astype(float), y, raw_data


def load_crawling_data_bert():
    from transformers import BertTokenizer
    import pandas as pd

    df1 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_1.csv", index_col=[0])
    # df2 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_2.csv", index_col=[0])
    # df3 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_3.csv", index_col=[0])
    # df4 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_4.csv", index_col=[0])
    df5 = pd.read_csv("./data/crawling_data/reviewdata_total_labeled_5.csv", index_col=[0])
    # ???????????? ???????????? ?????? df??????, ????????? ????????? ??????
    df = pd.concat([df1[:30000], df5[:30000]])  # , df2[:30000], df3[:30000], df4[:30000],
    df = df.loc[:, ['total_score', 'tokenized_review']]
    df_dropped = df.dropna(axis=0)
    data = df_dropped.sample(frac=1).reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(s) for s in data['preprocessed_review']]


    

def data_for_pretrain():
    import pandas as pd
    # from konlpy.tag import Okt
    # okt = Okt()
    # from tensorflow.keras.preprocessing.text import Tokenizer
    # tokenizer = Tokenizer()
    from gensim.models import Word2Vec, FastText
    tokenized_data = []
    y_list = []
    import pandas as pd
    import gzip
    import pickle

    # ???????????? ?????? ????????? ????????? ?????? ?????? ??????

    print('Loading data for pretrain...')
    cleandata = pd.read_csv("cleandata_labeled_1+5.csv", encoding='utf-8')
    df_shuffled = cleandata.sample(frac=1).reset_index(drop=True)
    data = df_shuffled['tokenized_review']
    y = np.array(df_shuffled['total_score'], dtype='int64')
    raw_data = []

    num_classes = np.max(y)
    print(num_classes, 'classes')

    min_count = 10
    # model = FastText(sentences=data, window=5, min_count=min_count, workers=4, sg=1)
    model = FastText.load_fasttext_format('./cc.ko.300.bin.gz')
    print(f'Vectorizing sequence data with min_count {min_count}...')
    x = np.empty((300,), dtype='float32')
    nwords = 0.
    # GPU ver.
    idx_to_key = model.wv.index2word
    # local ver.
    # idx_to_key = model.wv.index_to_key
    # key_to_idx = model.wv.key_to_index
    index2word_set = set(idx_to_key)
    for idx in tqdm(range(len(data)), desc="Word2Vec"):
        featureVec = np.zeros((300,), dtype='float32')
        for word in data[idx]:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model.wv[word])
        is_all_zero = not np.any(featureVec)
        if is_all_zero:
            pass
        else:
            featureVec = np.divide(featureVec, nwords)
            x = np.append(x, featureVec, axis=0)
            y_list.append(y[idx])
            raw_data.append(df_shuffled['tokenized_review'][idx])
    y = np.array(y_list, dtype='int64')
    x = x[300:]
    x = x.reshape(y.size, -1)
    print(len(y), 'train sequences')
    print('X_DATA for pretrain:', x.shape)
    print('raw_review_data shape:', len(raw_data))
    print('Y_DATA for pretrain:', y.shape)

    return x.astype(float), y



def load_data(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fmnist':
        return load_fashion_mnist()
    elif dataset_name == 'usps':
        return load_usps()
    elif dataset_name == 'pendigits':
        return load_pendigits()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        return load_reuters()
    elif dataset_name == 'stl':
        return load_stl()
    elif dataset_name == 'imdb':
        return load_imdb()
    elif dataset_name == 'crawling_data':
        return load_crawling_data()
    elif dataset_name == 'crawling_data_fasttext':
        return load_crawling_data_fasttext()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)