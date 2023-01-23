from fastapi import FastAPI
import requests

import pandas as pd
from typing import Dict
from collections import defaultdict

from omegaconf import OmegaConf
from app.utils.NaverCrawl.navercrawl import naver_crawl
from app.utils.Bigkindscrawl import bigkinds_crawl
from app.utils.BERTopic.bertopic_model import bertopic_modeling
from app.utils.Extract_context import extract_context
from app.utils.One_sent_summarization import summary_one_sent
from app.utils.KorBertSum.src.extract_topk_summarization import extract_topk_summarization
import time
app = FastAPI()

# 뉴스 데이터프레임
app.news_df = None
app.topic_df = None

#크롤링부터 한줄요약까지
@app.post("/company_name/")
def request_crawl_news(company_name:str, date_gte:int,date_lte:int,news_num:int = 999) -> Dict:
    '''
    input:
        company_name(str): 검색어
        date_gte(int): 시작일
        date_lte(int): 종료일
        news_num(int): 검색 뉴스 수
    output:
        Dict{"topics_number"(str) : 토픽 번호
             "topics_text"(str) : 토픽 한줄요약}
    '''
    times=[0 for i in range(5)]
    times[0]= time.time()
    '''
    #1. 크롤링
    print("crawl news")
    #Naver 크롤링 news_df = naver_crawl(company_name,news_num)
    #BigKinds 크롤링
    news_df = bigkinds_crawl(company_name,date_gte,date_lte) # news_df = ['title','description','titleNdescription','URL','date']
    #news_df.to_csv("crwal_news.csv",index=False)
    #news_df.to_pickle("crwal_news.pkl")
    times[1] = time.time()
    
    #2. 전처리
    print("extract context")
    news_df = extract_context(news_df)
    #news_df.to_csv(f"{company_name}_{date_gte}_{date_lte}_crwal_news_context.csv",index=False)
    #news_df.to_pickle(f"{company_name}_{date_gte}_{date_lte}_crwal_news_context.pkl")
    times[2] = time.time()

    #3. 토픽 분류
    print("start divide topic")
    cfg = OmegaConf.load(f"./app/config/bertopic_config.yaml")
    news_df = bertopic_modeling(cfg, news_df)
    #news_df.to_csv(f"{company_name}_after_bertopic_with_num.csv",index=False)
    #news_df.to_pickle(f"{company_name}_after_bertopic_with_num.pkl")
    
    times[3] = time.time()
    #4. 한줄요약
    print("summary one sentence")
    topic_df = pd.DataFrame()
    #토픽번호에 맞는 데이터만 가져오기
    for topic_number in set(news_df['topic']):
        if topic_number == -1:
            continue
        now_news_df = news_df[news_df['topic']==topic_number]
        now_topic_df = summary_one_sent(topic_number,now_news_df)
        topic_df = pd.concat([topic_df,now_topic_df])
    #topic_df.to_csv(f"{company_name}_topic_one_sent.csv",index = False)
    #topic_df.to_pickle(f"{company_name}_topic_one_sent.pkl")

    times[4] = time.time()
    print("crwal_end")
    print(f'crawl : {times[1] - times[0]}\ncontext: {times[2]-times[1]}\n BERTtopic: {times[3]-times[2]}\n onesent: {times[4]-times[3]}')
    print(f'total time : {times[4]-times[0]} sec')
    '''
    app.news_df = pd.read_pickle("after_bertopic.pkl")
    app.topic_df = pd.read_pickle("topic_one_sent.pkl")
    #5. 한줄요약 반환 result df = ['topic','one_sent's] 
    return {'topic': list(app.topic_df['topic']),'one_sent':list(app.topic_df['one_sent'])}
    
# 문단요약
@app.get("/summary/{topic_number}")
def request_summary_news(topic_number):
    #토픽 뉴스 수집
    
    now_news_df = app.news_df[app.news_df['topic'] == topic_number]
    
    #전처리

    #문단요약
    summary_text =""
    print(now_news_df.index)
    '''
    for topk in extract_topk_summarization(now_news_df).loc[1,'topk']:
        idx,text = topk
        summary_text = summary_text + text
        '''
    print(summary_text)
    #return = 문단
    return {"summarization":summary_text}
    

# 뉴스 리스트 출력
@app.get("/news/{topic_number}")
def request_summary_news(topic_number):
    #토픽 뉴스 수집
    now_news_df = app.news_df[app.news_df['topic'] == int(topic_number)]
    #print(len(now_news_df))
    #return = [날짜, 언론사, 헤드라인, url]
    '''
    return {"date":['20220101'],
            "title":['s'],
            "url":['s']
            }
    '''
    return {"date":list(now_news_df['date']),
            "title":list(now_news_df['title']),
            "url":list(now_news_df['url'])
            }
    


# 키워드 출력
@app.get("/keyword/{topic_number}")
def request_keyword_news(topic_number:int):
    #토픽 뉴스 수집

    # 키워드 뽑기

    #return = {"keyword": [keyword1, keyword2, ...]}
    return {"ans": "ans"}