from pandas_parallel_apply import DataFrameParallel
import multiprocessing as mp
from multiprocessing import freeze_support
import pandas as pd

from preprocessing import Preprocess
from crawl import CrawlNews
from data_insert import DataInsert

if __name__ == '__main__':

    freeze_support()
    prepro = Preprocess()
    cw = CrawlNews()
    db_server = DataInsert(index_name='bigkinds_newsdata')
    db_server.delete()
    
    # TODO file config

    if db_server.check_index():

        print("해당 인덱스는 이미 존재함")
        # TODO 인덱스가 이미 존재하면 없는 날짜만 찾아서 크롤링

    else:
        
        # TODO 해당 인덱스가 존재 하지 않으면 아예 통채로 데이터를 다운로드하기
        cw.crawl_from_file()
        df = cw.get_news_dataframe()

        # df = pd.read_pickle('test.pkl')
        # df = df.loc[~df['URL'].isna()]
        # df = df.reset_index(drop=True)
        
        # 전처리 작업 수행
        dfp = DataFrameParallel(df, n_cores=16, pbar=True)
        df['제목'] = dfp['제목'].apply(prepro)
        df['본문'] = dfp['본문'].apply(prepro)

        # elastic search에 insert
        db_server.insert(df)