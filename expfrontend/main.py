import datetime
from collections import Counter
import json
import pandas as pd
import requests

import streamlit as st
from streamlit import session_state as state
from streamlit.components.v1 import html
from annotated_text import annotated_text
from streamlit_elements import elements, sync, event, mui, dashboard

from confirm_button_hack import cache_on_button_press
# 페이지 타이틀
st.set_page_config(page_title="News Summarization", layout="wide")

# css 세팅
with open("expfrontend/style.css") as source_css:
    st.markdown(f"<style>{source_css.read()}</style>", unsafe_allow_html=True)

#state 세팅
if "company_name" not in state:
    state.company_name = ""
if "before_company_name" not in state:
    state.before_company_name = ""
if "before_search_date" not in state:
    state.before_search_date = (datetime.date(2022, 12, 1),datetime.date(2022, 12, 15),)
if "options_category" not in state:
    state['options_category'] = ["정치", "경제", "사회", "문화", "국제", "지역", "스포츠", "IT_과학"]
if "options_sentiment" not in state:
    state['options_sentiment'] = ["긍정", "부정", "중립"]
if 'search_list' not in state:
    search_list = pd.read_csv("asset/autocomplete.csv", index_col=0)["name"]
    search_list.loc[0] = ""
    search_list.sort_index(inplace=True)
    state['search_list'] = search_list
if 'stock_name_list' not in state:
    state['stock_name_list'] = pd.read_csv("asset/name_code.csv", index_col=0)            

# 검색페이지
def search_page():
    st.markdown("<h1 style='text-align: center;'>NEWSUMMARY</h1>", unsafe_allow_html=True)
    search_contain = st.empty()
    news_contain = st.empty()
    
    page_buttons = []
    with search_contain.container():
        _, center, _ = st.columns([1, 8, 1])
        # 검색창 + 자동완성 기능
        center.selectbox(
            label="회사명 혹은 종목코드를 입력하세요.",
            options=state['search_list'],
            label_visibility="collapsed",
            key = 'company_name'
        )
        
        _, col0, col1, col2, col3, _ = st.columns([2, 6, 3.5, 3.5, 3, 2])
        # checkbox options for article sentiment
        col1.multiselect(
            "기사 감성 선택",
            ["긍정", "부정", "중립"],
            default=state['options_sentiment'],
            on_change=None,
            key="options_sentiment"
        )
        # checkbox options for article category
        col2.multiselect(
            "기사 카테고리 선택",
            ["정치", "경제", "사회", "문화", "국제", "지역", "스포츠", "IT_과학"],
            default=state['options_category'],
            on_change=None,
            key = 'options_category'
        )
        # 기간 검색창
        col3.date_input(
            "기간",
            value=state.before_search_date,
            label_visibility="collapsed",
            key="search_date",
        )
        _  , center, _ = st.columns([1, 8, 1])
        
        # 검색한 경우
        if (state.before_company_name != "" or state.company_name != "")and len(state.search_date) > 1:
            # 검색어나 검색기간이 바뀌면 new데이터 새로 받기
            if state.company_name != "" and (state.before_company_name != state.company_name or state.before_search_date != state.search_date):
                state.before_company_name = state.company_name
                state.before_search_date = state.search_date
                start_date = f"{state.search_date[0].year:0>4d}{state.search_date[0].month:0>2d}{state.search_date[0].day:0>2d}"  # 시작검색일
                end_date = f"{state.search_date[1].year:0>4d}{state.search_date[1].month:0>2d}{state.search_date[1].day:0>2d}"  # 종료검색일
                # 회사이름 검색 요청
                # response = requests.post(f"http://localhost:8001/company_name/?company_name={company_name}&date_gte={start_date}&date_lte={end_date}&news_num=9999")
                # response = response.json()
                # news_df = pd.read_json(response["news_df"],orient="records")
                # topic_df = pd.read_json(response["topic_df"],orient="records")
                news_df = pd.read_pickle("expfrontend/news_df.pkl")
                topic_df = pd.read_pickle("expfrontend/topic_df2.pkl")
                state["news_df"] = news_df
                state["topic_df"] = topic_df

            # 뉴스가 없으면 결과가 없다고 반환
            if len(state["news_df"]) == 0:
                st.warning("검색 결과가 없습니다.", icon="⚠️")

            # 종목코드로 검색한 경우
            if state.before_company_name.isdigit():
                stock_num = state.before_company_name
                company_name = state['stock_name_list'][state['stock_name_list']["code"] == int(state.before_company_name)]['name'].values[0]
            # 회사명으로 검색한 경우
            else:
                stock_num = state['stock_name_list'][state['stock_name_list']["name"] == str(state.before_company_name)]["code"].values[0]
                stock_num = f"{int(stock_num):06}"
                company_name = state.before_company_name
            with col0:
                stock_wiget(stock_num)

            # 뉴스 요약 정보
            col3.info(
                f"""
                📰 검색된 뉴스 {len(state["news_df"])}개  
                🍪 추출 토픽 수 {len(state["topic_df"])}개 
                """
            )  # 🔥
            sentiment_dict = {'긍정':'positive', '중립':'neutral', '부정':'negative'}
            label_to_icon = {"negative": "😕", "neutral": "😐", "positive": "😃"}
            sentiment_color = {'positive':'#4593E7', 'negative':'#E52828', 'neutral':'#21E146'}

            # 선택된 카테고리만을 포함하도록 필터링
            topic_df_filtered = state['topic_df'].loc[state['topic_df']['category1'].isin(state['options_category'])]
            # 선택된 감성만 포함하도록 필터링
            options_sentiment = [sentiment_dict[i] for i in state['options_sentiment']]
            topic_df_filtered = topic_df_filtered.loc[topic_df_filtered['sentiment'].isin(options_sentiment)]
            # sort by category
            category1_sort_list = list(Counter(topic_df_filtered['category1']).keys())
            if '경제' in category1_sort_list: 
                category1_sort_list.remove('경제')
                category1_sort_list = ['경제'] + category1_sort_list
            
            for cat1 in category1_sort_list:
                now_topic_df = topic_df_filtered[topic_df_filtered['category1'] == cat1]
                now_topic_df = now_topic_df.sort_values(by=['sentiment'],ascending=False).reset_index(drop=False)
                cols = [0,0]
                _, cols[0], cols[1], _ = st.columns([1, 4, 4, 1])
                for idx, row in now_topic_df.iterrows():
                    topic_number = int(row["topic"])
                    topic_keyword = row["keyword"].split("_")

                    page_buttons.append(topic_number)
                    now_idx = idx % 2
                    with cols[now_idx]:
                        annotated_text(
                            (row["category1"], "Category", "#D1C9AC"),
                            (f"{label_to_icon[row['sentiment']]}", "Sentiment", sentiment_color[row["sentiment"]])
                            #f"{label_to_icon[topic_sentiment]}"
                            # (topic_keyword[4], "", "#8A9BA7"),
                        )
                        annotated_text(
                            (topic_keyword[0], "", "#B4C9C7"),
                            (topic_keyword[1], "", "#F3BFB3"),
                            (topic_keyword[2], "", "#F7E5B7"),
                            # (topic_keyword[4], "", "#8A9BA7"),
                        )
                    cols[now_idx].button(row["one_sent"], key=topic_number)
        else:
            empty1, center, empty2 = st.columns([0.9, 8, 0.9])
            with center:
                index_wiget()

    # 요약문 누르면 해당 페이지로
    for button_key in page_buttons:
        if state[button_key]:
            search_contain.empty()
            with news_contain.container():
                news_page(button_key)


# 뉴스 요약 페이지
def news_page(idx):
    # 한줄요약(제목)
    topics_text = state["topic_df"]["one_sent"][idx]
    topic_number = int(state["topic_df"]["topic"][idx])
    empty0 = st.write("")

    empty1, center, empty2 = st.columns([1, 8, 1])
    center.subheader(topics_text)
    empty1, _, col2, empty2 = st.columns([1, 7, 1, 1])
    back_button = col2.button("back")
    if back_button:
        page_buttons.clear()
        news_contain.empty()

    # 뉴스링크 [date,title,url]
    news_df = state["news_df"]
    news_list = news_df[news_df["topic"] == topic_number]
    news_list = news_list.reset_index(drop=True)

    empty1, center, empty2 = st.columns([1, 8, 1])
    empty1, col1, col2, empty2 = st.columns([0.6, 1, 5, 0.1])
    with center.expander("뉴스 링크"):
        for _, row in news_list[:12].iterrows():
            # empty1, col1, col2, empty2 = st.columns([0.6, 1, 5, 0.1])
            # col1, col2 = st.columns([1, 5])
            # st.text(row["date"])
            st.caption(f"<p>{row['date']} &nbsp&nbsp&nbsp&nbsp <a href='{row['url']}'>{row['title']}</a> </p>", unsafe_allow_html=True)

    # 요약문
    empty1, center, empty2 = st.columns([1, 8, 1])
    center.subheader("요약문")
    now_news_df = news_list[["context"]]
    now_news_json = now_news_df.to_json(orient="columns", force_ascii=False)
    # summarization = requests.post(f"http://localhost:8001/summary/",json=now_news_json)
    # summary_text = summarization.json()["summarization"]
    summary_text = """
    삼성전자가 3일 주주총회를 개최하고 유명희 전 산업부 통상교섭본과 허은녕 서울대 공대 사외이사 선임을 의결했다.

    삼성전자는 3일 주주총회를 열고 허은녕 서울대 공대 와 유명희 전 산업부 통상교섭본과 산업통상자원부 통상교섭본을 사외이사로 선임했다.

    2017년부터 2019년까지 학회 부회장을 지냈으며, 한국혁신학회 회장과 학회 회장 등을 지낸 에너지 부문의 석학으로 손꼽힌다.

    삼성전자는 3일 용인 삼성인재개발원에서 임시 주주총회를 열고 유명희 전 산업통상자원부 통상교섭본과 허은녕 서울대 공대 를 사외이사로 선임했다. 이재용 회장 취임 이후 열린 이번 임시주총에 대해 내년 3월 정기주총에서 이 회장을 등기이사로 선임하기 위한 사전작업이 아니냐는 분석이 나오고 있다.

    삼성전자는 3일 사외이사 선임으로 견제 및 감시기능이 강화되면서 삼성전자의 경영 투명성 확보와 소액주주 보호 역할이 확대될 것으로 평가했다.

    지난 3월 주주총회 이후 사외이사 4명, 사내이사 5명으로 이사회를 운영해 오던 삼성전자가 5월 박병국 사외이사가 5월 별세하고 한화진 사외이사가 새 정부의 초대 환경부 직을 맡으면서 사임해 6명의 사외이사 중 결원 2명이 생겼다고 밝히며 임시 주주총회를 개최했다. 학회 부회장, 한국혁신학회 회장, 학회 회장을 역임한 에너지 전문가인 맛허 사외이사는 전문가다.
    """
    center.write(summary_text)
    # 키워드
    center.subheader("키워드")


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
    "width": "100%",
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
