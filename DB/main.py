from typing import Optional
from fastapi import FastAPI

from Database.data_insert import DataInsert
from utils import NewsCluster, RemoveDup

import pickle

app = FastAPI()
DI = DataInsert('bigkinds_newsdata')
newscluster = NewsCluster()
removedup = RemoveDup()

@app.get("/")
async def root():
    return {"message": "Financial News dataset server"}

# get은 pydantic으로 안되나 본데
@app.get('/search/{query}')
def search_news(query_sentence: str,
                date_gte: Optional[int] = None,
                date_lte: Optional[int] = None,
                topk: Optional[int] = 9999,
                cluster : Optional[bool] = False,
                num_clusters : Optional[int] = 10):

    query_result = DI.search(query_sentence=query_sentence, date_gte=date_gte, date_lte=date_lte, topk=topk)

    if cluster:
        output = newscluster.cluster(data = query_result['hits']['hits'], num_clusters=num_clusters)
        return output
    else:
        return query_result['hits']['hits']

@app.get('/RemoveDup')
def remove_dup(data):

    with open("tmp.pickle","rb") as fr:
        data = pickle.load(fr)

    tmp_list = []
    for each in data:
        tmp_list.append(each['_source'])

    df = pd.DataFrame(tmp_list)
    tmp_df = df.loc[df.label == 3]
    title_N_desc = tmp_df['titleNdescription'].tolist()

    output = removedup.remove_dup(title_N_desc)

    return output

# TODO
# 클러스터
# 중복제거