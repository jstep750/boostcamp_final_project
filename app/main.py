from fastapi import FastAPI
import requests

import pandas as pd
from typing import Dict
from collections import defaultdict

from omegaconf import OmegaConf
from app.utils.NaverCrawl.navercrawl import naver_crawl
from app.utils.Bigkindscrawl import bigkinds_crawl
from app.utils.BERTopic.bertopic_model import bertopic_modeling
app = FastAPI()
# 뉴스 데이터프레임
news_df = defaultdict(list)

#크롤링부터 한줄요약까지
@app.get("/company_name/")
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
    #1. 크롤링
    #Naver 크롤링
    news_df = naver_crawl(company_name,news_num)
    #BigKinds 크롤링
    #news_df = bigkinds_crawl(company_name,date_gte,date_lte)

    #2. 전처리

    #3. 토픽 분류
    cfg = OmegaConf.load(f"./app/config/bertopic_config.yaml")
    #news_df = pd.read_csv("./app/utils/BERTopic/crawl_result(삼성전자).csv")
    news_df = bertopic_modeling(cfg, news_df)
    print(news_df.columns, len(news_df))

    topic_df = pd.DataFrame(columns=['topic_number','topic_text'])
    
    #4. 한줄요약
    #토픽번호에 맞는 데이터만 가져오기
    for topic_number in set(news_df['Topic']):
        if topic_number == -1:
            continue
        now_news_df = news_df[news_df['Topic']==topic_number]
        

    #5. 한줄요약 반환 result = ['topic1','topic2',....]
    topics_number=[0,1,2]
    
    topics_text = ["편의점 GS25가 '원스피리츠'와 협업해 선보인 원소주 스피릿이 지난해 GS25에서 판매되는 모든 상품 중 매출 순위 7위를 기록했다고 17일 밝혔다.", 
                    "원소주 스피릿은 출시 직후 2달 동안 입고 물량이 당일 완판되는 오픈런 행렬이 이어져 왔으며 최근 GS25와 원스피리츠의 공급 안정화 노력에 따라 모든 점포에서 수량제한 없이 상시 구매가 가능해졌다.", 
                    "GS25는 오는 18일 원소주 스피릿 누적 판매량 400만 병 돌파 기념으로 상시 운영되는 1개입 전용 패키지를 선보여 상품의 프리미엄을 더하기로 했다."]
    return {
        "topics_number": topics_number,
        "topics_text": topics_text}
    
# 문단요약
@app.get("/summary/{topic_number}")
def request_summary_news(topic_number:int):
    #토픽 뉴스 수집

    #전처리

    #문단요약

    #return = 문단
    return {"summarization":'''
            편의점 GS25가 '원스피리츠'와 협업해 선보인 원소주 스피릿이 지난해 GS25에서 판매되는 모든 상품 중 매출 순위 7위를 기록했다고 17일 밝혔다.\n
            원소주 스피릿은 출시 직후 2달 동안 입고 물량이 당일 완판되는 오픈런 행렬이 이어져 왔으며 최근 GS25와 원스피리츠의 공급 안정화 노력에 따라 모든 점포에서 수량제한 없이 상시 구매가 가능해졌다.\n
            GS25는 오는 18일 원소주 스피릿 누적 판매량 400만 병 돌파 기념으로 상시 운영되는 1개입 전용 패키지를 선보여 상품의 프리미엄을 더하기로 했다.
            '''}
    

# 뉴스 리스트 출력

@app.get("/news/{topic_number}")
def request_summary_news(topic_number:int):
    #토픽 뉴스 수집

    #return = [날짜, 언론사, 헤드라인, url]
    return {"date":["2023.01.16", "2023.01.16.","2023.01.17."],
            "press":["서울경제","헤럴드경제","머니투데이" ],
            "headline":["포스코건설, 설 맞아 협력사 거래대금 897억원 조기 지급", 
                        "연간 5만달러 넘는 해외송금 쉬워진다", 
                        "박재범 '원소주 스피릿', GS25서 400만 병 팔렸다"],
            "URL":["https://n.news.naver.com/mnews/article/011/0004145322?sid=101","https://n.news.naver.com/mnews/article/016/0002091309?sid=101","https://n.news.naver.com/mnews/article/008/0004841028?sid=101"]
            }

# 키워드 출력
@app.get("/keyword/{topic_number}")
def request_keyword_news(topic_number:int):
    #토픽 뉴스 수집

    # 키워드 뽑기

    #return = {"keyword": [keyword1, keyword2, ...]}
    return {"ans": "ans"}