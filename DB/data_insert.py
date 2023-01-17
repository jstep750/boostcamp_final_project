from elasticsearch import Elasticsearch
import json

# tmp use
import pandas as pd
from tqdm import tqdm

# TODO Elastic search security need to be updated
import warnings
warnings.filterwarnings(action='ignore')

class DataInsert:

    def __init__(self, index_name):
        
        self.es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10, retry_on_timeout=True)
        self.index_name = index_name
        self.setting_path = './config/setting.json'
    
    def check_index(self):
        
        if self.es.indices.exists(index=self.index_name):
            return True
        else:
            self._create_index()


    def _create_index(self):

        index_name = self.index_name

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        
        with open(self.setting_path, "r") as f:
            setting = json.load(f)

        self.es.indices.create(index=index_name, body=setting)

        print(f"{self.index_name} index has been successfully created")

    def insert(self, df):

        for each in tqdm(df.iterrows()):
            
            try:
                doc = {
                    'title': each[1]['제목'],
                    'description' : each[1]['본문'],
                    'titleNdescription' : ' '.join([each[1]['제목'], each[1]['본문']]),
                    'URL' : each[1]['URL'],
                    'date' : each[1]['일자'],
                }
                self.es.index(index=self.index_name, body=doc)

            except:
                pass

    
    def search(self, query, date_gte=20230116, date_lte=20230116, topk=9999):
        
        query_doc = {
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                    "match": {
                                        "titleNdescription": query
                                        }
                                    },
                                    {
                                        "range":{
                                            "date" : {
                                                "gte":date_gte,
                                                "lte":date_lte,
                                            }
                                        }
                                    }
                                ]
                            },
                        },
                    }

        res = self.es.search(index=self.index_name, body=query_doc, size=topk)

        return res

    def delete(self):

        try:
            self.es.indices.delete(index=self.index_name)
            return True
        except:
            print("No index_name to delete")
    
if __name__ == '__main__':

    DI = DataInsert('bigkinds_newsdata')

    # DI.create_index()
    # df = pd.read_excel('../NewsData/NewsResult_20230116-20230116.xlsx')
    # DI.insert(df)
    # df2 = pd.read_excel('../NewsData/NewsResult_20230115-20230115.xlsx')
    # DI.insert(df2)

    result = DI.search(query='삼성전자', date_gte=20230110, date_lte=20230115)
    
    print(len(result['hits']['hits']))
    breakpoint()
    # 여기서 뭘 더 해야하냐?

    # TODO
    # 데이터를 전부다 삽입을 하고
    # 백엔드를 만들고
    # 본문을 불러와야하나?
    # 쿼리를 불러 오면 중복을 제거를 하자

    # elastic search도 들어갔는데 못핡 없을듯? 본문을 그냥 아예 넣을까?