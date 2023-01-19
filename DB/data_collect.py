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

    if db_server.check_index():

        print("해당 인덱스는 이미 존재함")
    else:
        
        cw.crawl_from_file()
        df = cw.get_news_dataframe()

        breakpoint()
        dfp = DataFrameParallel(df, n_cores=16, pbar=True)
        df['제목'] = dfp['제목'].apply(prepro)
        df['본문'] = dfp['본문'].apply(prepro)
        df = df.loc[~df['URL'].isna()]

        breakpoint()

        # TODO elastic search insert
        print(df)