import time

import pandas as pd
import warnings
from pandas_parallel_apply import DataFrameParallel

import multiprocessing as mp
from multiprocessing import freeze_support
from newspaper import Article

from preprocess import Preprocess
import warnings
warnings.filterwarnings(action='ignore')

def crawl_news(url):

    if 'http://' in str(url) or 'https://' in str(url):
        pass
    else:
        url = 'https://' + str(url)

    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        return article.text
    except:
        return None

if __name__ == '__main__':
    freeze_support()

    new_data = pd.read_excel('./data/skhynix.xlsx')
    print(new_data.URL.head(5))

    prepro = Preprocess()

    start = time.time()
    dfp = DataFrameParallel(new_data, n_cores=16, pbar=True)
    new_data['context'] = dfp['URL'].apply(crawl_news)

    dfp = DataFrameParallel(new_data, n_cores=16, pbar=True)
    new_data['context'] = dfp['context'].apply(prepro)
    end = time.time()

    print(new_data.to_csv('test3.csv'))
    print(end - start)

# start = time.time()
# dfp = DataFrameParallel(new_data, n_cores=4, pbar=True)
# new_data['context'] = dfp['URL'].apply(lambda x : crawl_news(x))
# end = time.time()

# print(new_data['context'])
# print(end - start)

# for_test = 'https://view.asiae.co.kr/article/2022121811575100379'

# article = Article(for_test, language='ko')
# article.download()
# article.parse()

# prepro = Preprocess()
# result = prepro(article.text)

# print(result)