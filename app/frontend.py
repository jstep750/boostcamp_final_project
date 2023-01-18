import requests
import datetime 

import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press

#페이지 타이틀
st.set_page_config(page_title="News Summarization")
st.session_state.company_name=None
page_buttons=[]

#검색페이지
def search_page():
    #Google처럼 어플 제목으로 하는 것이 좋을듯
    st.markdown("<h1 style='text-align: center;'>News Summarization</h1>", unsafe_allow_html=True)
    search_contain = st.empty()
    news_contain = st.empty()
   
    with search_contain.container():
        #텍스트박스를 html로 다르게 바꿀 수 있을지도?
        col1, col2 = st.columns([2, 1])
        company_name = col1.text_input("검색",st.session_state.company_name if st.session_state.company_name else ""
                                , placeholder ="회사명 입력")
        st.session_state.company_name = company_name

        search_date = col2.date_input("기간",value=(datetime.datetime.now(), datetime.datetime.now()))
        start_date = f"{search_date[0].year:0>4d}{search_date[0].month:0>2d}{search_date[0].day:0>2d}"
        end_date = f"{search_date[1].year:0>4d}{search_date[1].month:0>2d}{search_date[1].day:0>2d}"
        if company_name:
            # 회사이름 검색 요청
            response = requests.get(f"http://localhost:8001/company_name/?company_name={company_name}&date_gte={start_date}&date_lte={end_date}")
            
            st.session_state["topic_number"] = response.json()['topics_number']
            st.session_state["topics_text"] = response.json()['topics_text']
            for topic_number, topic_text in zip(st.session_state["topic_number"],st.session_state["topics_text"]):
                page_buttons.append(st.button(topic_text))
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
    back_button = st.button("back")
    if back_button:
        page_buttons.clear()
        news_contain.empty()

page_names_to_funcs = {
    "search_page": search_page,
    "news_page": news_page
}
if __name__ == '__main__':
    search_page()
        
        