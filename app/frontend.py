import requests
import datetime 

import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press

#페이지 타이틀
st.set_page_config(page_title="News Summarization")
page_buttons=[]

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
    with search_contain.container():
        #검색창
        company_name = st.text_input("검색", placeholder ="회사명 입력",label_visibility='collapsed')
        st.session_state.company_name = company_name if company_name else st.session_state.company_name
        #기간 검색창
        col1, col2 = st.columns([5, 2])
        search_date = col2.date_input("기간",value=(datetime.datetime.now(), datetime.datetime.now()),label_visibility='collapsed')
        start_date = f"{search_date[0].year:0>4d}{search_date[0].month:0>2d}{search_date[0].day:0>2d}"  #시작검색일
        end_date = f"{search_date[1].year:0>4d}{search_date[1].month:0>2d}{search_date[1].day:0>2d}"    #종료검색일

        if st.session_state.company_name:
            # 회사이름 검색 요청
            if "topic_number" not in st.session_state:
                st.text("start")
                response = requests.get(f"http://localhost:8001/company_name/?company_name={st.session_state.company_name}&date_gte={start_date}&date_lte={end_date}&news_num=999").json()
                st.text("end")

                st.session_state["topic_number"] = response['topics_number']
                st.session_state["topics_text"] = response['topics_text']

            cols = st.columns([1,1])
            for i in range(2):
                cols[i].button(f"{st.session_state.company_name}{i}번째")
            
            for topic_number, topic_text in zip(st.session_state["topic_number"],st.session_state["topics_text"]):
                page_buttons.append(st.button(topic_text,key=f"button_{topic_number}"))
            
    for idx, button in enumerate(page_buttons):
        if button:
            with news_contain.container():
                news_page(idx)
            search_contain.empty()
    

#뉴스 요약 페이지
def news_page(idx):
    #한줄요약(제목)
    topics_text = st.session_state["topics_text"][idx]
    topic_number = st.session_state["topic_number"][idx]
    st.subheader(topics_text)

    #뉴스링크 [날짜,언론사,헤드라인,URL]
    news_list = requests.get(f"http://localhost:8001/news/{topic_number}").json()
    with st.expander("뉴스 링크"):
        for idx in range(len(news_list['date'])):
            col1, col2, col3 = st.columns([1,1,5])
            col1.text(news_list['date'][idx])
            col2.text(news_list['press'][idx])
            col3.caption(f"<a href='{news_list['URL'][idx]}'>{news_list['headline'][idx]}</a>",unsafe_allow_html=True)
        

    
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

page_names_to_funcs = {
    "search_page": search_page,
    "news_page": news_page
}
if __name__ == '__main__':
    search_page()
        
        