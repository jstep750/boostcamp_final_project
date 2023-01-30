import requests
import datetime 
import json
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press

import re

#페이지 타이틀
st.set_page_config(page_title="News Summarization",layout = 'wide')
#스크롤
st.markdown("""
                <html>
                    <head>
                    <style>
                        ::-webkit-scrollbar {
                            width: 10px;
                            }

                            /* Track */
                            ::-webkit-scrollbar-track {
                            background: #f1f1f1;
                            }

                            /* Handle */
                            ::-webkit-scrollbar-thumb {
                            background: #888;
                            }

                            /* Handle on hover */
                            ::-webkit-scrollbar-thumb:hover {
                            background: #555;
                            }
                    </style>
                    </head>
                    <body>
                    </body>
                </html>
            """, unsafe_allow_html=True)

with open("style.css") as source_css:
        st.markdown(f"<style>{source_css.read()}</style>",unsafe_allow_html=True)

#검색페이지
def search_page():    
    #Google처럼 어플 제목으로 하는 것이 좋을듯
    st.markdown("<h1 style='text-align: center;'>NEWSUMMARY</h1>", unsafe_allow_html=True)
    search_contain = st.empty()
    news_contain = st.empty()
    if 'company_name' not in st.session_state:
        st.session_state.company_name = ""
    if 'before_company_name' not in st.session_state:
        st.session_state.before_company_name = ""
    if 'before_search_date' not in st.session_state:
        st.session_state.before_search_date = (datetime.date(2022,12,1), datetime.date(2022,12,15))
    
    page_buttons=[]
    with search_contain.container():
        #검색창
        company_name = st.text_input("검색", value=st.session_state['company_name'], placeholder ="회사명 입력",label_visibility='collapsed', key="company_name")
        #기간 검색창
        _,col1,col2 = st.columns([15,1,2])
        search_date = col2.date_input("기간",value=st.session_state.before_search_date,label_visibility='collapsed', key = "search_date")
        
        #news_num = col1.number_input("뉴스 개수",0, 999,999,label_visibility='collapsed',key = "news_num")
        
        if st.session_state.company_name != "" and len(search_date) > 1 :   #검색했으면
            #검색어나 검색기간이 바뀌면 new데이터 새로 받기
            if st.session_state.before_company_name != st.session_state.company_name or st.session_state.before_search_date !=st.session_state.search_date:
                st.session_state.before_company_name = st.session_state.company_name
                st.session_state.before_search_date = st.session_state.search_date
                
                start_date = f"{st.session_state.search_date[0].year:0>4d}{st.session_state.search_date[0].month:0>2d}{st.session_state.search_date[0].day:0>2d}"  #시작검색일
                end_date = f"{st.session_state.search_date[1].year:0>4d}{st.session_state.search_date[1].month:0>2d}{st.session_state.search_date[1].day:0>2d}"    #종료검색일
                # 회사이름 검색 요청
                #response = requests.post(f"http://localhost:8001/company_name/?company_name={st.session_state.company_name}&date_gte={start_date}&date_lte={end_date}&news_num=9999")
                #response = response.json()
                #news_df = pd.read_json(response["news_df"],orient="records")
                #topic_df = pd.read_json(response["topic_df"],orient="records")
                news_df = pd.read_pickle("news_df.pkl")
                topic_df = pd.read_pickle("topic_df.pkl")
                st.session_state["news_df"] = news_df
                st.session_state["topic_df"] = topic_df
            #뉴스가 없으면 결과가 없다고 반환
            if len(st.session_state["news_df"]) == 0:
                st.warning('검색 결과가 없습니다.', icon="⚠️")
            
            #dataframe 보기            
            #st.write(st.session_state["news_df"])
            #st.write(st.session_state["topic_df"])
            
            #버튼 추가  
            label_to_icon = {"negative":"😕","neutral":"😐","positive":"😃"}
            col1, col2 = st.columns([1,1])
            max_idx = len(st.session_state["topic_df"]) 
            for idx in range(max_idx):
                topic_sentiment = st.session_state["topic_df"]["sentiment"][idx]
                topic_number = st.session_state["topic_df"]["topic"][idx]
                topic_text = st.session_state["topic_df"]["one_sent"][idx]
                page_buttons.append(idx)
                if idx%2 == 0:
                    col1.button(label_to_icon[topic_sentiment] + topic_text,key=idx)
                else:
                    col2.button(label_to_icon[topic_sentiment] + topic_text,key=idx)

    # 요약문 누르면 해당 페이지로
    for button_key in page_buttons:
        if st.session_state[button_key]:
            search_contain.empty()
            with news_contain.container():
                news_page(button_key)
    

#뉴스 요약 페이지
def news_page(idx):
    #한줄요약(제목)
    topics_text = st.session_state["topic_df"]["one_sent"][idx]
    topic_number = int(st.session_state["topic_df"]["topic"][idx])
    st.subheader(topics_text)
    _, col2 = st.columns([7,1])
    back_button = col2.button("back")
    if back_button:
        page_buttons.clear()
        news_contain.empty()

    #뉴스링크 [date,title,url]
    news_df = st.session_state["news_df"]
    news_list = news_df[news_df['topic'] == topic_number]
    news_list = news_list.reset_index(drop=True)
    with st.expander("뉴스 링크"):
        for _, row in news_list[:12].iterrows():
            col1, col2 = st.columns([1,5])
            col1.text(row['date'])
            col2.caption(f"<a href='{row['url']}'>{row['title']}</a>",unsafe_allow_html=True)    
   
    #요약문
    st.subheader("요약문")
    now_news_df = news_list[['context']]
    now_news_json = now_news_df.to_json(orient = "columns",force_ascii=False)
    #summarization = requests.post(f"http://localhost:8001/summary/",json=now_news_json)
    #summary_text = summarization.json()["summarization"]
    summary_text = """
    삼성전자가 3일 주주총회를 개최하고 유명희 전 산업부 통상교섭본과 허은녕 서울대 공대 사외이사 선임을 의결했다.

    삼성전자는 3일 주주총회를 열고 허은녕 서울대 공대 와 유명희 전 산업부 통상교섭본과 산업통상자원부 통상교섭본을 사외이사로 선임했다.

    2017년부터 2019년까지 학회 부회장을 지냈으며, 한국혁신학회 회장과 학회 회장 등을 지낸 에너지 부문의 석학으로 손꼽힌다.

    삼성전자는 3일 용인 삼성인재개발원에서 임시 주주총회를 열고 유명희 전 산업통상자원부 통상교섭본과 허은녕 서울대 공대 를 사외이사로 선임했다. 이재용 회장 취임 이후 열린 이번 임시주총에 대해 내년 3월 정기주총에서 이 회장을 등기이사로 선임하기 위한 사전작업이 아니냐는 분석이 나오고 있다.

    삼성전자는 3일 사외이사 선임으로 견제 및 감시기능이 강화되면서 삼성전자의 경영 투명성 확보와 소액주주 보호 역할이 확대될 것으로 평가했다.

    지난 3월 주주총회 이후 사외이사 4명, 사내이사 5명으로 이사회를 운영해 오던 삼성전자가 5월 박병국 사외이사가 5월 별세하고 한화진 사외이사가 새 정부의 초대 환경부 직을 맡으면서 사임해 6명의 사외이사 중 결원 2명이 생겼다고 밝히며 임시 주주총회를 개최했다. 학회 부회장, 한국혁신학회 회장, 학회 회장을 역임한 에너지 전문가인 맛허 사외이사는 전문가다.
    """
    st.write(summary_text)
    #키워드
    st.subheader("키워드")
    

if __name__ == '__main__':
    search_page()
        
        