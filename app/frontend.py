import requests
import datetime 

import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press

#페이지 타이틀
st.set_page_config(page_title="News Summarization",layout="wide")
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

def button_click(idx):
    st.wirte(idx)

#검색페이지
def search_page():
    
    #Google처럼 어플 제목으로 하는 것이 좋을듯
    st.markdown("<h1 style='text-align: center;'>NEWSUMMARY</h1>", unsafe_allow_html=True)
    search_contain = st.empty()
    news_contain = st.empty()
    with open("app/style.css") as source_css:
        st.markdown(f"<style>{source_css.read()}</style>",unsafe_allow_html=True)
    if 'company_name' not in st.session_state:
        st.session_state.company_name = ""
    page_buttons=[]
    with search_contain.container():
        #검색창
        company_name = st.text_input("검색", value=st.session_state['company_name'], placeholder ="회사명 입력",label_visibility='collapsed', key="company_name")
        #기간 검색창
        col1, col2 = st.columns([5, 2])
        search_date = col2.date_input("기간",value=(datetime.date(2022,12,26), datetime.date(2022,12,30)),label_visibility='collapsed')
        if len(search_date) > 1:
            start_date = f"{search_date[0].year:0>4d}{search_date[0].month:0>2d}{search_date[0].day:0>2d}"  #시작검색일
            end_date = f"{search_date[1].year:0>4d}{search_date[1].month:0>2d}{search_date[1].day:0>2d}"    #종료검색일

        if st.session_state.company_name != "":
            # 회사이름 검색 요청
            response = requests.get(f"http://localhost:8001/company_name/?company_name={st.session_state.company_name}&date_gte={start_date}&date_lte={end_date}&news_num=999")
            response = response.json()
            st.session_state["topic_number"] = response['topic']
            st.session_state["topics_text"] = response['one_sent']
            #버튼 추가   
            for idx in range(int(len(st.session_state["topic_number"]) / 2)):                
                col1, col2 = st.columns([1,1])
                topic_number = st.session_state["topic_number"][idx * 2]
                topic_text = st.session_state["topics_text"][idx * 2]
                if len(topic_text) > 60:
                    topic_text = topic_text[0:60] + "..."
                col1.button(topic_text,key=idx * 2)
                page_buttons.append(idx * 2)

                topic_number = st.session_state["topic_number"][idx * 2 + 1]
                topic_text = st.session_state["topics_text"][idx * 2 + 1]
                if len(topic_text) > 60:
                    topic_text = topic_text[0:60] + "..."
                col2.button(topic_text,key=idx * 2 + 1)
                page_buttons.append(idx * 2 + 1)
                
            
            if len(st.session_state["topic_number"]) % 2 == 1:
                col1, col2 = st.columns([1,1])
                topic_number = st.session_state["topic_number"][-1]
                topic_text = st.session_state["topics_text"][-1]
                if len(topic_text) > 60:
                    topic_text = topic_text[0:60] + "..."
                col1.button(topic_text,key=len(st.session_state["topic_number"]) -1)
                page_buttons.append(len(st.session_state["topic_number"])-1)
    
    for button_key in page_buttons:
        if st.session_state[button_key]:
            with news_contain.container():
                news_page(button_key)
            search_contain.empty()    
    st.write(st.session_state)

#뉴스 요약 페이지
def news_page(idx):
    #한줄요약(제목)
    topics_text = st.session_state["topics_text"][idx]
    topic_number = int(st.session_state["topic_number"][idx])
    st.subheader(topics_text)
    
    #뉴스링크 [date,title,URL]
    news_list = requests.get(f"http://localhost:8001/news/{topic_number}").json()
    with st.expander("뉴스 링크"):
        for news_idx in range(len(news_list['date'])):
            col1, col2 = st.columns([1,5])
            col1.text(news_list['date'][news_idx])
            col2.caption(f"<a href='{news_list['url'][news_idx]}'>{news_list['title'][news_idx]}</a>",unsafe_allow_html=True)
        
    #요약문
    st.subheader("요약문")
    summarization = requests.get(f"http://localhost:8001/summary/{topic_number}")
    st.write(summarization.json()["summarization"])
    #키워드
    st.subheader("키워드")
    _, col2 = st.columns([7,1])
    back_button = col2.button("back")
    if back_button:
        page_buttons.clear()
        news_contain.empty()
    

if __name__ == '__main__':
    search_page()
        
        