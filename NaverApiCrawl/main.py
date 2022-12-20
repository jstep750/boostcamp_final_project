import os
import sys
import urllib.request

import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title('Naver news crawl')
query = st.text_input("검색어를 입력해주세요")

if query:

    headers = {
        "X-Naver-Client-Id":"GV0sLpuZoaE73nEjOyc1",
        "X-Naver-Client-Secret":"uD20MiNcbQ"
    }

    data = {
        'query':query, # 검색어
        'display':10, # 검색 개수 default : 10, max : 100
        'start':1, # 검색 시작 위치 default : 1, max : 100
        'sort':'sim' # 검색 결과 정렬 방법 : sim 정확도순, date 날짜순
    }

    data = urllib.parse.urlencode(data)

    result = requests.get("https://openapi.naver.com/v1/search/news.json?", headers=headers, params=data)

    print(result.json())
    crawl_result = result.json()['items']
    crawl_result = pd.DataFrame(crawl_result)

    #crawl_result.to_csv('crawl_result.csv')
    st.table(crawl_result)