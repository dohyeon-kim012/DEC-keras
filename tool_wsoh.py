import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
okt = Okt()
import pickle
import gzip

# csv 파일 읽기
def csv_reader(file_name):
    # 경로 설정
    csv_location = "./data/{name}.csv".format(name=file_name)
    # 로컬 > data_frame > 변수에 할당후 결과 확인
    read_data = pd.read_csv(csv_location, index_col=[0])
    read_data = read_data.reset_index(drop=True)

    return read_data

# csv 파일 저장 함수
def csv_save(data_name, file_name):

    # 경로 설정
    csv_location = "./data/{name}.csv".format(name= file_name)

    # data > data_frame > csv 저장
    crwled_data_frame = pd.DataFrame(data_name)
    crwled_data_frame.to_csv(csv_location, encoding= 'utf-8')



# 형태소 분석 함수
def okt_morph(dataframe):
    # 범위 지정 가능
    df_pre = dataframe['preprocessed_review']
    df_corpus = df_pre[:]

    clean_words = []
    for i in tqdm(df_corpus):
        ok = okt.pos(i)
        # ok = okt.pos(i, stem=True)

        words = []
        for word in ok:
            if word[1] in ['Adjective', 'Verb', 'Noun', 'Adverb', 'Exclamation', 'Determiner', 'Unknown']:
                # 감정을 표현하는 품사만 따로 가져 오는 것이 유용하다고 판단.
                # 필요에 따라 추가시키기에도 좋다.

                # Exclamation : 감탄사는 감정의 가장 단순한 형태여서 꼭 필요하다고 생각
                # Determiner : 새호텔, 헌호텔 과 같은 리뷰가 점수에 영향이 있을것이라고 판단
                # Unknown : 미등록어는 널리쓰이는 유행어 같은 것을 놓치지 않기 위해 선택
                words.append(word[0])
        clean_words.append(words)

    return clean_words



# 문장에서 단어 벡터의 평균을 구하는 함수
def make_feat_vec(words, model, num_features):

    # 0으로 채운 배열로 초기화 한다(속도향상을 위해)
    feature_vec = np.zeros((num_features,), dtype="float32")

    nwords = 0
    # index2word는 모델 사전에 있는 단어명을 담은 리스트
    # 속도 향상을 위해 set 형태로 초기화
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어면 피쳐에 추가
    for word in words:
        if word in index2word_set:
            nwords += 1
            feature_vec = np.add(feature_vec, model[word])

    # 결과를 단어수로 나누어 평균을 구한다.
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec



# okt 형태소 분류 자료 csv merge
def okt_csv(file_name):

    df_all = csv_reader(file_name)  # csv 파일 load
    df_preprocess = df_all.loc[:, ['review_id', 'score', 'review', 'preprocessed_review']]  # 점수, 전처리된 리뷰, 리뷰id(merge할때 필요)
    df_clean = df_preprocess.dropna(axis=0)  # nan 값이 있는 행 삭제
    df_reindex = df_clean.reset_index(drop=True)  # 인덱스 재정렬

    x, y = df_reindex['preprocessed_review'], df_reindex['score']  # x 리뷰, y 점수

    print('okt 형태소 분류...')  # 형태소 분류
    df_x1 = pd.DataFrame(x)  # 전처리 리뷰 데이터 프레임 변환
    tokenized_riviews = okt_morph(df_x1)  # 전처리된 리뷰 데이터 토크나이징

    dict_okt = {'okt_pos_review': tokenized_riviews}
    df_okt = pd.DataFrame(dict_okt)
    df_merge = pd.merge(df_reindex, df_okt, right_index=True, left_index=True)
    df_merge.to_csv(f"./data/{file_name}_Okt_version.csv", encoding='utf-8')


# okt 리뷰데이터 파싱 함수
def reviews_parcing(file_name):
    df_all = csv_reader(file_name)  # csv 파일 load
    df_preprocess = df_all.loc[:, ['total_score', 'tokenized_review']]  # 점수와 전처리된 리뷰만 가져옴
    df_clean = df_preprocess.dropna(axis=0)  # nan 값이 있는 행 삭제
    df_reindex = df_clean.reset_index(drop=True)  # 인덱스 재정렬
    okt_review, label = df_reindex['tokenized_review'], df_reindex['total_score']

    x_list = okt_review.values.tolist()

    input_final = []
    for i in tqdm(range(len(x_list)), desc= 'okt 리뷰 data >>> pkl list data'):
        x_list[i] = x_list[i].replace("[", "")
        x_list[i] = x_list[i].replace("]", "")
        x_list[i] = x_list[i].replace(",", "")
        x_list[i] = x_list[i].replace("'", "")
        x_list[i] = x_list[i].split()

        input_final.append(x_list[i])

    return input_final


# 리스트 파일로 저장(피클)
def save_list(list, save_name):
    with gzip.open(f'{save_name}.pkl', 'wb') as f:
        pickle.dump(list, f, pickle.HIGHEST_PROTOCOL)

# 리스트 파일 불러오기(피클)
def load_list(save_name):
    with gzip.open(f'{save_name}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def csv_reindex(origin_file_name):

    csv_location = "./data/{name}.csv".format(name=origin_file_name)
    read_data = pd.read_csv(csv_location, index_col=[0])
    df_reindex = read_data.reset_index(drop=True)
    df_reindex.to_csv(csv_location, encoding='utf-8')


if __name__ == "__main__":

    # files = ['review_data_by_total_score_1',
    #          'review_data_by_total_score_2',
    #          'review_data_by_total_score_3',
    #          'review_data_by_total_score_4',
    #          'review_data_by_total_score_5']

    # 데이터 랜덤 섞기
    '''
    df_all = csv_reader('test_review_data_5000')
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    csv_save(df_all, 'test_review_data_5000')    
    '''

    # kobert용 데이터 만들기
    '''
    df_all = csv_reader('test_review_data_5000')
    
    # label number change(cuda 오류 : 클래스?의 시작이 1번이면 오류가 난다)
    df_all.loc[(df_all.total_score == 1), 'total_score'] = 0
    df_all.loc[(df_all.total_score == 2), 'total_score'] = 1
    df_all.loc[(df_all.total_score == 3), 'total_score'] = 2
    df_all.loc[(df_all.total_score == 4), 'total_score'] = 3
    df_all.loc[(df_all.total_score == 5), 'total_score'] = 4

    # 랜덤하게 섞기
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    # 리뷰 추출
    df_train = df_all.loc[:3999, ['review_id', 'preprocessed_review', 'total_score']]
    df_test = df_all.loc[4000:, ['review_id', 'preprocessed_review', 'total_score']]

    df_train = df_train.dropna(subset=['total_score'])
    df_train = df_train.reset_index(drop=True)

    df_test = df_test.dropna(subset=['total_score'])
    df_test = df_test.reset_index(drop=True)

    csv_save(df_train, 'train_data_4000')
    csv_save(df_test, 'test_data_1000')
    '''

    # 리뷰 파싱
    '''
    for file_name in files:
        list_reviews = reviews_parcing(file_name)
        save_list(list_reviews, f'pkl_list_{file_name}')
    '''

    # test data 만들기
    '''
    df_score_1 = csv_reader(files[0])
    df_score_2 = csv_reader(files[1])
    df_score_3 = csv_reader(files[2])
    df_score_4 = csv_reader(files[3])
    df_score_5 = csv_reader(files[4])

    # 랜덤하게 섞기
    df_score_1 = df_score_1.sample(frac=1).reset_index(drop=True)
    df_score_2 = df_score_2.sample(frac=1).reset_index(drop=True)
    df_score_3 = df_score_3.sample(frac=1).reset_index(drop=True)
    df_score_4 = df_score_4.sample(frac=1).reset_index(drop=True)
    df_score_5 = df_score_5.sample(frac=1).reset_index(drop=True)

    # 리뷰 추출
    df_score_1_selected = df_score_1.iloc[:2500, :]
    df_score_2_selected = df_score_2.iloc[:1999, :]
    df_score_3_selected = df_score_3.iloc[:1999, :]
    df_score_4_selected = df_score_4.iloc[:1999, :]
    df_score_5_selected = df_score_5.iloc[:2500, :]

    # df_score_merged = pd.concat([df_score_1_selected,
    #                              df_score_5_selected])

    df_score_merged = pd.concat([df_score_1_selected,
                                 df_score_2_selected,
                                 df_score_3_selected,
                                 df_score_4_selected,
                                 df_score_5_selected])

    df_score_merged = df_score_merged.dropna(subset=['total_score'])
    df_score_merged = df_score_merged.reset_index(drop=True)
    csv_save(df_score_merged, 'review_data_1_5_5000')
    '''

    # 한개의 파일만 파싱
    '''
    list_reviews = reviews_parcing('test_review_data_5000')
    save_list(list_reviews, 'pkl_list_test_review_data_5000')
    '''

    # total_review score 별로 csv 저장
    '''
    df_all_reviews = csv_reader('all_hotels_review_data(dropna)')

    for score in range(1,6):
        score_condition = (df_all_reviews.total_score == score)
        df_selected_by_score = df_all_reviews[score_condition]
        df_selected_by_score = df_selected_by_score.reset_index(drop=True)

        csv_save(df_selected_by_score, f'review_data_by_total_score_{score}')
    '''

    # 가장 긴 리뷰 길이 구하기
    '''
    list_pkl = load_list('review_list')
    all_sentence_len_list = []
    for list_each in tqdm(list_pkl, desc="리뷰 길이 구하기"):
        list_each_len = len(list_each)
        all_sentence_len_list.append(list_each_len)
    
    max_len = max(all_sentence_len_list)
    '''

    # 전체리뷰(600만개) score 분포 구하기
    '''
    df_review_data_all = csv_reader('cluster_50_review_data_cluster_num_32')

    print('** 모든 리뷰 데이터 score 분포 **')
    print('=============================')
    print(df_review_data_all['total_score'].value_counts())
    '''

    # 클러스터 num 을 선택하여 저장하기
    '''
    df_cluster_all = csv_reader('cluster_50_review_data')
    cluster_numbers = [16, 32, 24]
    for cluster_number in cluster_numbers:
        cluster_condition = (df_cluster_all.cluster_num == cluster_number)
        df_cluster_each = df_cluster_all[cluster_condition]
        csv_save(df_cluster_each, f'cluster_50_review_data_cluster_num_{cluster_number}')
    '''

    # 클러스터링 통계 구해 보기
    '''
    cluster_num = 5
    for num in range(cluster_num):
        df_test_by_cluster_num = csv_reader(f'cluster_{num}')
        print(f'** cluster_num : {num} **')
        print('** score 분포 **')
        print('==================')
        print(df_test_by_cluster_num['score'].value_counts())
        print('==================')
    '''
    '''
    df_test_for_mincnt = csv_reader('cluster_test_score0_score5')
    print('각 스코어 개수')
    print(df_test_for_mincnt['total_score'].value_counts())
    print('각 클러스터 개수')
    print(df_test_for_mincnt['cluster_num'].value_counts())
    '''


    '''
    # 조건 설정
    cluster_0_condition = (df_cluster_all.cluster_num == 0) # & (df_cluster_all.score == 1)
    cluster_1_condition = (df_cluster_all.cluster_num == 1)
    cluster_2_condition = (df_cluster_all.cluster_num == 2)
    cluster_3_condition = (df_cluster_all.cluster_num == 3)
    cluster_4_condition = (df_cluster_all.cluster_num == 4)

    # 조건에 맞는 데이터 출력 및 저장
    df_cluster_0 = df_cluster_all[cluster_0_condition]
    df_cluster_1 = df_cluster_all[cluster_1_condition]
    df_cluster_2 = df_cluster_all[cluster_2_condition]
    df_cluster_3 = df_cluster_all[cluster_3_condition]
    df_cluster_4 = df_cluster_all[cluster_4_condition]

    csv_save(df_cluster_0, 'cluster_0')
    csv_save(df_cluster_1, 'cluster_1')
    csv_save(df_cluster_2, 'cluster_2')
    csv_save(df_cluster_3, 'cluster_3')
    csv_save(df_cluster_4, 'cluster_4')
    '''

    # 데이터 합치기
    '''
    df1 = csv_reader('cluster_0')
    df2 = csv_reader('cluster_12')
    df3 = csv_reader('cluster_9')
    df4 = csv_reader('cluster_18')
    df5 = csv_reader('cluster_5')
    df6 = csv_reader('cluster_16')

    df_merge = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

    csv_save(df_merge, 'SCORE_1_TEST_DATA_30000_3')
    # print(df_merge)
    '''

    # 클러스터 num 으로 구분하여 csv 저장후 자료 살펴보기
    file_name = 'test_labeled_1_2ndtry_5'
    df_cluster_all = csv_reader(file_name)

    cluster_num = 31
    # for num in cluster_num:
    cluster_condition = (df_cluster_all.cluster_num == cluster_num)
    df_cluster_each = df_cluster_all[cluster_condition]
    csv_save(df_cluster_each, f'maindata_{file_name}')

    # 하위 클러스터 데이터 따로 모으기
    # 전체 데이터 프레임에서 cluster_num이 cluster_number에 있는 경우 삭제
    # for cluster_number in cluster_num:
    cluster_delete_condition = df_cluster_all[df_cluster_all['cluster_num'] == cluster_num].index
    df_cluster_all = df_cluster_all.drop(cluster_delete_condition, axis=0)
    df_cluster_all = df_cluster_all.reset_index(drop=True)
    csv_save(df_cluster_all, f'rest_{file_name}')

    # df1 = csv_reader('cluster_0')
    # df2 = csv_reader('cluster_6')
    # df3 = csv_reader('cluster_9')
    # # df4 = csv_reader('cluster_18')
    # # df5 = csv_reader('cluster_5')
    # # df6 = csv_reader('cluster_16')
    #
    # df_merge = pd.concat([df1, df2, df3], ignore_index=True)
    #
    # csv_save(df_merge, 'SCORE_1_TEST_DATA_30000_5')




