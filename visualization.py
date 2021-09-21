import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(color_codes=True)
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = [9, 6]

# result.csv 파일의 정확도 시각화
df_csv = pd.read_csv('./DEC-keras/results/exp1/results.csv', header=0, index_col=0)
'''print(df_csv)
                       acc      nmi      ari
trials                                      
usps                   NaN      NaN      NaN
0                  0.69692  0.67976  0.57756
1                  0.70026  0.67527  0.57742
2                  0.71284  0.68219  0.58093
3                  0.70918  0.68225  0.58835
...                    ...      ...      ...
6                  0.49247      0.0      0.0
7                  0.49247      0.0      0.0
8                  0.49247      0.0      0.0
9                  0.49247      0.0      0.0
        0.4924700000000001      0.0      0.0

[141 rows x 3 columns]'''

result = df_csv.loc['crawling_data':, 'acc']
'''print(result)
trials
crawling_data                   NaN
0                           0.49247
1                           0.49247
2                           0.49247
3                           0.49247
4                           0.49247
5                           0.49247
6                           0.49247
7                           0.49247
8                           0.49247
9                           0.49247
                 0.4924700000000001
0                           0.49247
1                           0.49247
2                           0.49247
3                           0.49247
4                           0.49247
5                           0.49247
6                           0.49247
7                           0.49247
8                           0.49247
9                           0.49247
                 0.4924700000000001
Name: acc, dtype: object'''
sns.barplot(x=, y='acc', data=result, palette='Blues_d')
plt.show()

### 연구 더 필요함