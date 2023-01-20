from elasticsearch import Elasticsearch
import json

import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

class DataInsert:
    """ Elasticsearch is used as a database for news search

    Attributes:
        self.es (Elasticsearch) : Elasticsearch instance
        self.index_name (str) : index name of news dataset repository
        self.setting_path (str) : Path of Elastic Search settings.json file

    Returns:
        None
    """
    def __init__(self, cfgs):

        self.es = Elasticsearch(
            "http://localhost:9200",
            request_timeout=30,
            max_retries=10,
            retry_on_timeout=True,
        )
        self.index_name = cfgs.data_insert_args.index_name
        self.setting_path = cfgs.data_insert_args.setting_path

    def check_index(self):
        """ Checking index whether index_name exists

        when self.index_name exists do nothing and return True
        when self.index_name does not exist create one and return True if successful
        when something go wrong return False

        Args:

        Returns:
            True if successful, False otherwise.

        """

        try:
            if not self.es.indices.exists(index=self.index_name):
                self._create_index()
            return True
        except:
            return False

    def _create_index(self):
        """ Creating an index in Elasticsearch

        create an index based on self.index_name args 

        """

        index_name = self.index_name

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)

        with open(self.setting_path, "r") as f:
            setting = json.load(f)

        self.es.indices.create(index=index_name, body=setting)

        print(f"{self.index_name} index has been successfully created")


    def insert(self, df):

        """ Checking index whether index_name exists

        when self.index_name exists do nothing and return True
        when self.index_name does not exist create one and return True if successful
        when something go wrong return False

        Args:

        Returns:
            True if successful, False otherwise.

        """

        for each in tqdm(df.iterrows()):

            try:
                doc = {
                    "title": each[1]["제목"],
                    "description": each[1]["본문"],
                    "titleNdescription": " ".join([each[1]["제목"], each[1]["본문"]]),
                    "URL": each[1]["URL"],
                    "date": each[1]["일자"],
                }
                self.es.index(index=self.index_name, body=doc)

            except:
                pass

    def search(self, query_sentence, date_gte=20230115, date_lte=20230116, topk=9999):

        query_doc = {
            "bool": {
                "must": [
                    {"match": {"titleNdescription": query_sentence}},
                    {
                        "range": {
                            "date": {
                                "gte": date_gte,
                                "lte": date_lte,
                            }
                        }
                    },
                ]
            },
        }

        res = self.es.search(index=self.index_name, query=query_doc, size=topk)

        return res

    def return_maxdate(self):

        query_doc = {
            "size": 0,
            "aggs": {
                "doc_with_max_run_id": {
                    "top_hits": {"sort": [{"date": {"order": "desc"}}], "size": 1}
                }
            },
        }

        res = self.es.search(index=self.index_name, body=query_doc)

        # fmt : off
        return res["aggregations"]["doc_with_max_run_id"]["hits"]["hits"][0]["_source"]["date"]

    def delete(self):

        try:
            self.es.indices.delete(index=self.index_name)
            return True
        except:
            print("No index_name to delete")


if __name__ == "__main__":

    DI = DataInsert("bigkinds_newsdata")
    result = DI.return_maxdate()
    print(result)

    # result = DI.search("삼성전자")

    # print(result)
