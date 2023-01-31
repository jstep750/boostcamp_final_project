import requests
import datetime
import json
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press
from streamlit_elements import elements, mui
import streamlit_elements
import re
from collections import Counter

# 페이지 타이틀
st.set_page_config(page_title="News Summarization", layout="wide")
stock_name_list = pd.read_csv("name_code.csv", index_col=0)
search_list = pd.read_csv("autocomplete.csv", index_col=0)["name"]
search_list.loc[0] = ""
search_list.sort_index(inplace=True)

with open("frontend/style.css") as source_css:
    st.markdown(f"<style>{source_css.read()}</style>", unsafe_allow_html=True)

# 검색페이지
def search_page():
    st.markdown("<h1 style='text-align: center;'>NEWSUMMARY</h1>", unsafe_allow_html=True)
    search_contain = st.empty()
    news_contain = st.empty()
    if "company_name" not in st.session_state:
        st.session_state.company_name = ""
    if "before_company_name" not in st.session_state:
        st.session_state.before_company_name = ""
    if "before_search_date" not in st.session_state:
        st.session_state.before_search_date = (
            datetime.date(2022, 12, 1),
            datetime.date(2022, 12, 15),
        )
    page_buttons = []
    with search_contain.container():
        # 검색창
        # company_name = st.text_input(
        #     "검색",
        #     value=st.session_state["company_name"],
        #     placeholder="회사명 혹은 종목코드를 입력해주세요.",
        #     label_visibility="collapsed",
        #     key="company_name",
        # )

        # 자동완성 기능
        company_name = st.selectbox(
            label="회사명 혹은 종목코드를 입력하세요.",
            options=search_list,
            label_visibility="collapsed",
        )

        # 기간 검색창
        _, col1, col2 = st.columns([15, 1, 2])
        search_date = col2.date_input(
            "기간",
            value=st.session_state.before_search_date,
            label_visibility="collapsed",
            key="search_date",
        )

        # news_num = col1.number_input("뉴스 개수",0, 999,999,label_visibility='collapsed',key = "news_num")

        # 검색어 입력하기 전에는 지수 정보 display
        if not (company_name != "" and len(search_date) > 1):
            index_wiget()
        # 검색한 경우
        elif company_name != "" and len(search_date) > 1:
            # 종목코드로 검색한 경우
            if company_name.isdigit():
                stock_num = company_name
                st.session_state["company_name"] = stock_name_list.iloc[stock_name_list[stock_name_list["code"] == int(company_name)].index]["name"].values[0]
            # 회사명으로 검색한 경우
            else:
                stock_num = stock_name_list.iloc[stock_name_list[stock_name_list["name"] == str(company_name)].index]["code"].values[0]
                stock_num = f"{int(stock_num):06}"
                st.session_state["company_name"] = company_name

            print(stock_num)
            print(company_name)
            print(st.session_state.company_name)

            stock_wiget(stock_num)

            # 검색어나 검색기간이 바뀌면 new데이터 새로 받기
            if st.session_state.before_company_name != st.session_state.company_name or st.session_state.before_search_date != st.session_state.search_date:
                st.session_state.before_company_name = st.session_state.company_name
                st.session_state.before_search_date = st.session_state.search_date

                start_date = f"{st.session_state.search_date[0].year:0>4d}{st.session_state.search_date[0].month:0>2d}{st.session_state.search_date[0].day:0>2d}"  # 시작검색일
                end_date = f"{st.session_state.search_date[1].year:0>4d}{st.session_state.search_date[1].month:0>2d}{st.session_state.search_date[1].day:0>2d}"  # 종료검색일
                # 회사이름 검색 요청
                response = requests.post(f"http://localhost:8001/company_name/?company_name={st.session_state.company_name}&date_gte={start_date}&date_lte={end_date}&news_num=9999")
                response = response.json()
                news_df = pd.read_json(response["news_df"],orient="records")
                topic_df = pd.read_json(response["topic_df"],orient="records")
                st.session_state["news_df"] = news_df
                st.session_state["topic_df"] = topic_df
            # 뉴스가 없으면 결과가 없다고 반환
            if len(st.session_state["news_df"]) == 0:
                st.warning("검색 결과가 없습니다.", icon="⚠️")

            cat1_cnt = Counter(topic_df['hard_category1']).most_common(2)
            most_cat1 = cat1_cnt[0][0] if cat1_cnt[0][0] != '경제' else cat1_cnt[1][0]   
            cate_df_list = list()
            cate_df_list.append(topic_df[topic_df['hard_category1']=='경제'])
            cate_df_list.append(topic_df[topic_df['hard_category1']==most_cat1])
            cate_df_list.append(topic_df[(topic_df['hard_category1'] != '경제') & (topic_df['hard_category1'] != most_cat1)])
            
            #버튼 추가  
            label_to_icon = {"negative":"😕","neutral":"😐","positive":"😃"}
            cate_label = ['경제',most_cat1,'기타']
            for idx in range(3):
                now_cate_label = cate_label[idx]
                col1, col2 = st.columns([1,15])
                col1.markdown(f"<div class='test' sytle = 'align-items : center; margin: 0 auto;'>{now_cate_label}</div>", unsafe_allow_html=True)
                for _, row in cate_df_list[idx].iterrows():
                    topic_number = int(row['topic'])
                    topic_text = row['one_sent']
                    topic_sentiment = row['sentiment']
                    col2.button(label_to_icon[topic_sentiment] + topic_text,key=topic_number)
                st.write("---")

    # 요약문 누르면 해당 페이지로
    for button_key in page_buttons:
        if st.session_state[button_key]:
            search_contain.empty()
            with news_contain.container():
                news_page(button_key)


# 뉴스 요약 페이지
def news_page(idx):
    # 한줄요약(제목)
    topics_text = st.session_state["topic_df"]["one_sent"][idx]
    topic_number = int(st.session_state["topic_df"]["topic"][idx])
    st.subheader(topics_text)
    _, col2 = st.columns([7, 1])
    back_button = col2.button("back")
    if back_button:
        page_buttons.clear()
        news_contain.empty()

    # 뉴스링크 [date,title,url]
    news_df = st.session_state["news_df"]
    news_list = news_df[news_df["topic"] == topic_number]
    news_list = news_list.reset_index(drop=True)
    with st.expander("뉴스 링크"):
        for _, row in news_list[:12].iterrows():
            col1, col2 = st.columns([1, 5])
            col1.text(row["date"])
            col2.caption(f"<a href='{row['url']}'>{row['title']}</a>", unsafe_allow_html=True)

    # 요약문
    st.subheader("요약문")
    now_news_df = news_list[["context"]]
    now_news_json = now_news_df.to_json(orient="columns", force_ascii=False)
    summarization = requests.post(f"http://localhost:8001/summary/",json=now_news_json)
    summary_text = summarization.json()["summarization"]
    st.write(summary_text)
    # 키워드
    st.subheader("키워드")


def index_wiget():
    html(
        """
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-tickers.js" async>
        {
        "symbols": [
        {
        "description": "KOSPI",
        "proName": "KRX:KOSPI"
        },
        {
        "description": "KOSDAQ",
        "proName": "KRX:KOSDAQ"
        },
        {
        "description": "NASDAQ 100",
        "proName": "NASDAQ:NDX"
        },
        {
        "description": "S&P 500",
        "proName": "FRED:SP500"
        },
        {
        "description": "USD/KRW",
        "proName": "FX_IDC:USDKRW"
        }
        ],
        "colorTheme": "light",
        "isTransparent": false,
        "showSymbolLogo": true,
        "locale": "kr"
        }
        </script>
        </div>
        <!-- TradingView Widget END -->
        """
    )


def stock_wiget(stock_num):
    info = """
    "symbol": "KRX:{0}",
    "width": "55%",
    "height": "100%",
    "locale": "kr",
    "dateRange": "3M",
    "colorTheme": "light",
    "trendLineColor": "rgba(255, 0, 0, 1)",
    "underLineColor": "rgba(204, 0, 0, 0.3)",
    "underLineBottomColor": "rgba(41, 98, 255, 0)",
    "isTransparent": false,
    "autosize": false,
    "largeChartUrl": ""
    """.format(
        stock_num
    )
    info = "{" + info + "}"

    docstring = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <div class="tradingview-widget-copyright"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
    {0}
    </script>
    </div>
    <!-- TradingView Widget END -->
    """.format(
        info
    )
    html(docstring)


if __name__ == "__main__":
    search_page()
