import requests
import datetime
import json
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from confirm_button_hack import cache_on_button_press

import re
from annotated_text import annotated_text

# 페이지 타이틀
st.set_page_config(page_title="News Summarization", layout="wide")
stock_name_list = pd.read_csv("name_code.csv", index_col=0)
search_list = pd.read_csv("autocomplete.csv", index_col=0)["name"]
search_list.loc[0] = ""
search_list.sort_index(inplace=True)

# 스크롤
st.markdown(
    """
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
            """,
    unsafe_allow_html=True,
)

with open("style.css") as source_css:
    st.markdown(f"<style>{source_css.read()}</style>", unsafe_allow_html=True)

# 검색페이지
def search_page():

    # Google처럼 어플 제목으로 하는 것이 좋을듯
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

        empty1, center, empty2 = st.columns([1, 8, 1])

        company_name = center.selectbox(
            label="회사명 혹은 종목코드를 입력하세요.",
            options=search_list,
            label_visibility="collapsed",
        )

        st.session_state["company_name"] = company_name

        # 기간 검색창
        empty1, col0, empty2, col1, col2, empty3 = st.columns([2, 6, 3.5, 3.5, 3, 2])
        # empty1, col1, empty2 = st.columns([13.5, 4.5, 2])

        # checkbox options for article sentiment
        with empty2:
            options_sentiment = st.multiselect(
                "기사 감성 선택",
                ["긍정", "부정", "중립"],
                default=["긍정", "부정", "중립"],
                on_change=None
            )

        sentiment_color = {'positive':'#4593E7', 'negative':'#E52828', 'neutral':'#21E146'}

        # checkbox options for article category
        with col1:
            options_category = st.multiselect(
                "기사 카테고리 선택",
                ["정치", "경제", "사회", "문화", "국제", "지역", "스포츠", "IT_과학"],
                default=["정치", "경제", "사회", "문화", "국제", "지역", "스포츠", "IT_과학"],
                on_change=None
            )

        #category_color = {'정치':'', '경제':'', '사회':'', '문화':'', '국제':'', '지역':'', '스포츠':'', 'IT_과학':''}

        search_date = col2.date_input(
            "기간",
            value=st.session_state.before_search_date,
            label_visibility="collapsed",
            key="search_date",
        )

        # news_num = col1.number_input("뉴스 개수",0, 999,999,label_visibility='collapsed',key = "news_num")

        # 검색어 입력하기 전에는 지수 정보 display
        if not (st.session_state.company_name != "" and len(search_date) > 1):
            empty1, center, empty2 = st.columns([0.9, 8, 0.9])
            with center:
                index_wiget()

        # 검색한 경우
        elif company_name != "" and len(search_date) > 1:
            empty0 = st.write("")
            # 종목코드로 검색한 경우
            if company_name.isdigit():
                stock_num = company_name
                st.session_state["company_name"] = stock_name_list.iloc[stock_name_list[stock_name_list["code"] == int(company_name)].index]["name"].values[0]
            # 회사명으로 검색한 경우
            else:
                stock_num = stock_name_list.iloc[stock_name_list[stock_name_list["name"] == str(company_name)].index]["code"].values[0]
                stock_num = f"{int(stock_num):06}"
                st.session_state["company_name"] = company_name

            with col0:
                stock_wiget(stock_num)

            # 검색어나 검색기간이 바뀌면 new데이터 새로 받기
            if st.session_state.before_company_name != st.session_state.company_name or st.session_state.before_search_date != st.session_state.search_date:
                st.session_state.before_company_name = st.session_state.company_name
                st.session_state.before_search_date = st.session_state.search_date

                start_date = f"{st.session_state.search_date[0].year:0>4d}{st.session_state.search_date[0].month:0>2d}{st.session_state.search_date[0].day:0>2d}"  # 시작검색일
                end_date = f"{st.session_state.search_date[1].year:0>4d}{st.session_state.search_date[1].month:0>2d}{st.session_state.search_date[1].day:0>2d}"  # 종료검색일
                # 회사이름 검색 요청
                # response = requests.post(f"http://localhost:8001/company_name/?company_name={st.session_state.company_name}&date_gte={start_date}&date_lte={end_date}&news_num=9999")
                # response = response.json()
                # news_df = pd.read_json(response["news_df"],orient="records")
                # topic_df = pd.read_json(response["topic_df"],orient="records")
                news_df = pd.read_pickle("news_df.pkl")
                topic_df = pd.read_pickle("topic_df2.pkl")
                st.session_state["news_df"] = news_df
                st.session_state["topic_df"] = topic_df

                # f'''뉴스 요약 정보:
                # 검색된 뉴스 {len(news_df)}개,
                # 추출 토픽 {len(topic_df)}개'''
                # summary_info = col2.info(''' ''')
                col2.info(
                    f"""
                    📰 검색된 뉴스 {len(news_df)}개  
                    🍪 추출 토픽 수 {len(topic_df)}개 
                    """
                )  # 🔥
            
            # 뉴스가 없으면 결과가 없다고 반환
            if len(st.session_state["news_df"]) == 0:
                st.warning("검색 결과가 없습니다.", icon="⚠️")

            # 선택된 카테고리만을 포함하도록 필터링
            st.session_state['topic_df_filtered'] = st.session_state['topic_df']
            st.session_state['topic_df_filtered'] = st.session_state['topic_df_filtered'].loc[st.session_state['topic_df_filtered']['category1'].isin(options_category)]
            
            # 선택된 감성만 포함하도록 필터링
            sentiment_dict = {'긍정':'positive', '중립':'neutral', '부정':'negative'}
            options_sentiment = pd.Series(options_sentiment).map(sentiment_dict).tolist()
            st.session_state['topic_df_filtered'] = st.session_state['topic_df_filtered'].loc[st.session_state['topic_df_filtered']['sentiment'].isin(options_sentiment)]
            
            # sentiment column에 색깔 mapping
            st.session_state['topic_df_filtered']['sentiment_color'] = st.session_state['topic_df_filtered']['sentiment'].map(sentiment_color)

            # sory by category
            st.session_state['topic_df_filtered'] = st.session_state['topic_df_filtered'].sort_values(by=['category1']).reset_index(drop=False)

            colors = ["#8ef", "#faa", "#afa", "#fea"]
            # 버튼 추가
            label_to_icon = {"negative": "😕", "neutral": "😐", "positive": "😃"}
            empty1, col1, col2, empty2 = st.columns([1, 4, 4, 1])
            max_idx = len(st.session_state["topic_df_filtered"])

            # topic_df => topic_df_filtered로 전부 교체
            for idx in range(max_idx):
                topic_sentiment = st.session_state["topic_df_filtered"]["sentiment"][idx]
                topic_number = st.session_state["topic_df_filtered"]["topic"][idx]
                topic_text = st.session_state["topic_df_filtered"]["one_sent"][idx]
                topic_keyword = st.session_state["topic_df_filtered"]["keyword"][idx].split("_")

                # 추가된 부분
                topic_category = st.session_state["topic_df_filtered"]["category1"][idx]
                topic_sentiment_color = st.session_state['topic_df_filtered']['sentiment_color'][idx]
                origin_idx = st.session_state['topic_df_filtered']['index'][idx]
                # 추가된 부분

                page_buttons.append(origin_idx)
                if idx % 2 == 0:
                    with col1:
                        annotated_text(
                            (topic_category, "Category", "#D1C9AC"),
                            (f"{label_to_icon[topic_sentiment]}", "Sentiment", topic_sentiment_color)
                            #f"{label_to_icon[topic_sentiment]}"
                            # (topic_keyword[4], "", "#8A9BA7"),
                        )
                    with col1:
                        annotated_text(
                            (topic_keyword[0], "", "#B4C9C7"),
                            (topic_keyword[1], "", "#F3BFB3"),
                            (topic_keyword[2], "", "#F7E5B7"),
                            # (topic_keyword[4], "", "#8A9BA7"),
                        )
                    col1.button(topic_text, key=origin_idx)
                    
                    

                else:
                    with col2:
                        annotated_text(
                            (topic_category, "Category", "#D1C9AC"),
                            (f"{label_to_icon[topic_sentiment]}", "Sentiment", topic_sentiment_color)
                            #f"{label_to_icon[topic_sentiment]}"
                            # (topic_keyword[4], "", "#8A9BA7"),
                        )
                    with col2:
                        annotated_text(
                            (topic_keyword[0], "", "#B4C9C7"),
                            (topic_keyword[1], "", "#F3BFB3"),
                            (topic_keyword[2], "", "#F7E5B7"),
                            # (topic_keyword[4], "", "#8A9BA7"),
                        )
                    col2.button(topic_text, key=origin_idx)

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
    empty0 = st.write("")

    empty1, center, empty2 = st.columns([1, 8, 1])
    center.subheader(topics_text)
    empty1, _, col2, empty2 = st.columns([1, 7, 1, 1])
    back_button = col2.button("back")
    if back_button:
        page_buttons.clear()
        news_contain.empty()

    # 뉴스링크 [date,title,url]
    news_df = st.session_state["news_df"]
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
