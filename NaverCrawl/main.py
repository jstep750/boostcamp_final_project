import os
import sys
#import urllib.request

import streamlit as st
import pandas as pd
import numpy as np
#import requests
import multiprocessing as mp
from multiprocessing import freeze_support

from pandas_parallel_apply import DataFrameParallel

from crawl import Crawl
from context import context
from preprocess import Preprocess

if __name__ == '__main__':
    freeze_support()

    st.title('Naver news crawl')

    query = st.text_input("검색어를 입력해주세요")
    number = st.text_input("검색할 숫자를 입력하세요")

    print("New Session start")

    if query and number:

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

        # result.to_csv('result3.csv')
        st.table(result[['title','context','pubDate']])