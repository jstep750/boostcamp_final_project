from typing import Optional
from fastapi import FastAPI

from Database.data_insert import DataInsert
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Financial News dataset server"}


@app.get('/search/{query}')
def search_news(query: str, date_gte: Optional[int] = None, date_lte: Optional[int] = None, topk: Optional[int] = 9999):

    DI = DataInsert('bigkinds_newsdata')
    result = DI.search(query=query, date_gte=date_gte, date_lte=date_lte, topk=topk)

    return result['hits']['hits']