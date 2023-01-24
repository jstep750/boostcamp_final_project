import requests
import datetime 

import json
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press

#페이지 타이틀
st.set_page_config(page_title="News Summarization",layout = 'wide')
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

with open("frontend/style.css") as source_css:
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
    page_buttons=[]
    with search_contain.container():
        #검색창-
        company_name = st.text_input("검색", value=st.session_state['company_name'], placeholder ="회사명 입력",label_visibility='collapsed', key="company_name")
        #기간 검색창
        _, col2 = st.columns([5,2])
        search_date = col2.date_input("기간",value=(datetime.date(2022,12,26), datetime.date(2022,12,30)),label_visibility='collapsed')
            
        if st.session_state.company_name != "" and len(search_date) > 1:
            start_date = f"{search_date[0].year:0>4d}{search_date[0].month:0>2d}{search_date[0].day:0>2d}"  #시작검색일
            end_date = f"{search_date[1].year:0>4d}{search_date[1].month:0>2d}{search_date[1].day:0>2d}"    #종료검색일

            if st.session_state.before_company_name != st.session_state.company_name:
                st.session_state.before_company_name = st.session_state.company_name
                # 회사이름 검색 요청
                response = requests.post(f"http://localhost:8001/company_name/?company_name={st.session_state.company_name}&date_gte={start_date}&date_lte={end_date}&news_num=999")
                response = response.json()
                news_df = pd.read_json(response["news_df"],orient="records")
                topic_df = pd.read_json(response["topic_df"],orient="records")
                st.session_state["news_df"] = news_df
                st.session_state["topic_df"] = topic_df
            
            #버튼 추가  
            max_idx = len(st.session_state["topic_df"]) 
            for idx in range(int(max_idx / 2)):                
                col1, col2 = st.columns([1,1])
                topic_number = st.session_state["topic_df"]["topic"][idx * 2]
                topic_text = st.session_state["topic_df"]["one_sent"][idx * 2]
                if len(topic_text) > 60:
                    topic_text = topic_text[0:60] + "..."
                col1.button(topic_text,key=idx * 2)
                page_buttons.append(idx * 2)

                topic_number = st.session_state["topic_df"]["topic"][idx * 2 + 1]
                topic_text = st.session_state["topic_df"]["one_sent"][idx * 2 + 1]
                if len(topic_text) > 60:
                    topic_text = topic_text[0:60] + "..."
                col2.button(topic_text,key=idx * 2 + 1)
                page_buttons.append(idx * 2 + 1)
            
            if max_idx % 2 == 1:
                col1, col2 = st.columns([1,1])
                topic_number = st.session_state["topic_df"]["topic"][max_idx - 1]
                topic_text = st.session_state["topic_df"]["one_sent"][max_idx - 1]
                if len(topic_text) > 60:
                    topic_text = topic_text[0:60] + "..."
                col1.button(topic_text,key=max_idx - 1)
                page_buttons.append(max_idx - 1)
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

    #뉴스링크 [date,title,URL]
    #news_list = requests.get(f"http://localhost:8001/news/{topic_number}").json()
    news_df = st.session_state["news_df"]
    news_list = news_df[news_df['topic'] == topic_number]
    with st.expander("뉴스 링크"):
        for _, row in news_list[:10].iterrows():
            col1, col2 = st.columns([1,5])
            col1.text(row['date'])
            col2.caption(f"<a href='{row['url']}'>{row['title']}</a>",unsafe_allow_html=True)
            
    
   
    #요약문
    st.subheader("요약문")
    summarization = requests.post(f"http://localhost:8001/summary/{topic_number}",now_news_df={news_df})
    st.write(summarization.json()["summarization"])
    #키워드
    st.subheader("키워드")
    
    
    

if __name__ == '__main__':
    search_page()
        
        