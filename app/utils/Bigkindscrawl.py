import pandas as pd
import requests
from collections import defaultdict

def bigkinds_crawl(company_name:str,date_gte:int,date_lte:int,topk:int=9999) -> pd.DataFrame:
    '''
    input:
        company_name(str): 검색어
        date_gte(int): 시작일
        date_lte(int): 종료일
        news_num(int): 검색 뉴스 수
    output:
        dataFrame["title", "titleNdescription", "context", "URL", "date", "category1", "category2"]
    '''

    #response = List[{'_index','_type','_id','_score', '_source':{'title','description','titleNdescription','URL','date'}}]
    response = requests.get(f"http://27.96.131.161:30001/new_search/{company_name}/?query_sentence={company_name}&index_name=bigkinds_new2&field=titleNdescription&date_gte={date_gte}&date_lte={date_lte}&topk={topk}").json()
    # 데이터프레임으로 변환 news_df = ['title','description','titleNdescription','URL','date']
    news_df = defaultdict(list)
    for idx in range(len(response)):
        response_source = response[idx]['_source']
        for key, value in response_source.items():
            news_df[key].append(value)
    news_df = pd.DataFrame(news_df)
    news_df = news_df.reset_index(drop=True)
    
    return news_df

if __name__ == "__main__":
    print(bigkinds_crawl("suo","20221201","20230120"))

