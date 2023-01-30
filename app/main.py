from fastapi import FastAPI, Response, Request
import requests

import json
import pandas as pd
from typing import Dict, Union, List
from collections import defaultdict
import time
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

from omegaconf import OmegaConf
from app.utils.Bigkindscrawl import bigkinds_crawl
from app.utils.BERTopic.bertopic_model import bertopic_modeling
from app.utils.One_sent_summarization import summary_one_sent
from app.utils.KorBertSum.src.extract_topk_summarization import extract_topk_summarization
from app.utils.KorBertSum.src.topic_summary import make_summary_paragraph

app = FastAPI()

def split_category_df(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    news_df를 category1 기준으로 "경제", "IT_과학", 그 외로 나누는 함수
    Args:
        news_df (pd.DataFrame): DB에서 response로 가져온 news_df

    Returns:
        pd.DataFrame: news_df를 economy_df, it_df, others_df로 나누어서 return
    """
    economy_df = news_df[(news_df["category1"] == "경제")]
    it_df = news_df[(news_df["category1"] == "IT_과학")]

    others_df = news_df[~((news_df["category1"] == "경제")|(news_df["category1"] == "IT_과학"))]
    return economy_df, it_df, others_df

#크롤링부터 한줄요약까지
@app.post("/company_name/")
def request_crawl_news(company_name:str, date_gte:int,date_lte:int,news_num:int = 999) -> Response:
    '''
    input:
        company_name(str): 검색어
        date_gte(int): 시작일
        date_lte(int): 종료일
        news_num(int): 검색 뉴스 수
    output:
        Dict{"news_df" : 뉴스 dataframe_tojson
             "topic_df" : 토픽 dataframe_tojson}
    '''
    times=[0 for i in range(4)]    
    times[0]= time.time()

    #1. 크롤링
    print("crawl news")
    news_df = bigkinds_crawl(company_name,date_gte,date_lte) # news_df = ['title','description','url','date']
    times[1] = time.time()
    #3. 토픽 분류
    print("start divide topic")
    #cfg = OmegaConf.load(f"./app/config/bertopic_config.yaml")
    news_df = bertopic_modeling(news_df)
    times[2] = time.time()

    #4. 한줄요약
    print("summary one sentence")
    topic_df = summary_one_sent(news_df)
    times[3] = time.time()

    print("crwal_end")
    print(f'crawl : {times[1] - times[0]}\n BERTtopic: {times[2]-times[1]}\n onesent: {times[3]-times[2]}')
    print(f'total time : {times[3]-times[0]} sec')
    
    #5. 한줄요약 반환
    result = json.dumps({"news_df": news_df.to_json(orient = "records",force_ascii=False) ,"topic_df": topic_df.to_json(orient = "records",force_ascii=False)})   
    return Response(result, media_type="application/json")
    
# 문단요약
@app.post("/summary/")
async def request_summary_news(request:Request): 
    body_bytes = await request.body()
    summary_text = ""
    if body_bytes:
        news_json = await request.json()
        now_news_df = pd.read_json(news_json,orient="columns")
        times=[0 for i in range(3)]    
        times[0]= time.time()
        #추출요약
        summary_df = extract_topk_summarization(now_news_df)
        times[1]= time.time()
        #생성요약
        summary_text = make_summary_paragraph(summary_df)
        times[2]= time.time()
        print(f"extract time : {times[1]-times[0]} sec \nparagraph time : {times[2]-times[1]} sec")
    return {"summarization":summary_text}
    
# 키워드 출력
@app.get("/keyword/{topic_number}")
def request_keyword_news(topic_number:int):
    #토픽 뉴스 수집

    # 키워드 뽑기

    #return = {"keyword": [keyword1, keyword2, ...]}
    return {"ans": "ans"}