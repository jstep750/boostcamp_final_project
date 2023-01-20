import pandas as pd
import numpy as np

import multiprocessing as mp
from multiprocessing import freeze_support
from pandas_parallel_apply import DataFrameParallel

import os
import sys
from pathlib import Path
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent, "")
sys.path.append(ASSETS_DIR_PATH)

from navercrawl_crawl import Crawl
from navercrawl_context import context
from navercrawl_preprocess import Preprocess


def naver_crawl(query:str,number:int = 999) -> pd.DataFrame:
    '''
    input:
        queary : 검색어
        number : 뉴스기사 수
    output:
        result : DataFrame['title', 'originallink', 'link', 'description', 'pubDate', 'context']
    '''
    freeze_support()

    prepro = Preprocess()

    print(' ### Api Call Start ###')
    cw = Crawl(client_id = "GV0sLpuZoaE73nEjOyc1", client_secret = "uD20MiNcbQ")
    result = cw(query=query, number=number)

    print(' ### Parsering start ###')
    dfp = DataFrameParallel(result, n_cores=16, pbar=True)
    result['context'] = dfp['link'].apply(context)

    print(' ### Preprocessing start ###')
    dfp = DataFrameParallel(result, n_cores=16, pbar=True)
    result['context'] = dfp['context'].apply(prepro)
    return result
    