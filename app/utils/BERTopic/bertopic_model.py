import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from konlpy.tag import Mecab
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ASSETS_DIR_PATH = os.path.join(Path(__file__).parent, "")
sys.path.append(ASSETS_DIR_PATH)


class CustomTokenizer:
    def __init__(self, tagger, stopwords):
        self.tagger = tagger
        self.stopwords = stopwords

    def __call__(self, sent):
        word_tokens = self.tagger.morphs(sent)
        result = [
            word for word in word_tokens if len(word) > 1 and word not in self.stopwords
        ]
        return result


def concat_title_context(df):
    add_title_context = []
    for _, t in df.iterrows():
        context = [t["title"]] + t["context"][0:2]
        add_title_context.append((" ".join(context)).strip())
    df["concat_text"] = add_title_context
    return df


def screened_articles(df, threshold=0.3):
    df = concat_title_context(df)
    indexes = []
    for topic_n in sorted(df["topic"].unique()):
        if topic_n == -1:
            continue
        matrix = []
        topic_df = df[df["topic"] == topic_n]
        idx = topic_df.index
        length = len(topic_df)
        # print(f'Num of articles on topic {topic_n}: {length}')
        for c1 in topic_df["concat_text"]:
            tmp = []
            for c2 in topic_df["concat_text"]:
                s1 = set(c1)
                s2 = set(c2)
                actual_jaccard = float(len(s1.intersection(s2))) / float(
                    len(s1.union(s2))
                )
                tmp.append(actual_jaccard)
            matrix.append(tmp)
        matrix = np.array(matrix)
        matrix = np.array(matrix > threshold)
        G = nx.from_numpy_array(matrix)
        attribute_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
        article_idx = [
            idx
            for (idx, _) in attribute_sorted
            if (attribute_sorted[0][1] - int(length * 0.1)) <= _
        ]
        if len(attribute_sorted) >= 3 and len(article_idx) < 3:
            article_idx = [idx for (idx, _) in attribute_sorted[:3]]
        elif len(article_idx) > 12:
            article_idx = [idx for (idx, _) in attribute_sorted[:12]]
        # print(f'Num of articles after screening: {len(article_idx)}')
        indexes += list(idx[article_idx])
    return df.iloc[indexes]


def getTitleNDescriptions(df: pd.DataFrame) -> List[str]:
    """
    pd.dataframe에서 title과 description 컬럼만 합쳐서 List로 반환하는 함수
    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        List[str]: title과 description를 합쳐서 List로 반환
    """
    docs = df["titleNdescription"].tolist()
    return docs

def bertopic_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bertopic modeling을 진행하는 함수

    Args:
        df (pd.DataFrame): 해당 쿼리에 대한 input dataframe
    Returns:
        pd.DataFrame: bertopic modeling을 통해 나온 topic column + input dataframe
    """
    cfg = OmegaConf.load(os.path.join(ASSETS_DIR_PATH, "bertopic_config.yaml"))
    file_cfg = cfg.file
    model_cfg = cfg.model

    docs = getTitleNDescriptions(df)

    file_path = os.path.join(Path(__file__).parent, file_cfg.stopwords_path)
    f = open(file_path, "r")
    ko_stop_words = [text.rstrip() for text in f.readlines()]

    custom_tokenizer = CustomTokenizer(Mecab(), ko_stop_words)
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

    # embedding 생성
    model_name = model_cfg.model_name
    sentence_model = SentenceTransformer(model_name, device='cuda')
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Create instances of GPU-accelerated UMAP and HDBSCAN
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=8,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=model_cfg.top_n_words,
        # nr_topics = model_cfg.nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    start = time.time()
    topics, probs = model.fit_transform(documents=docs, embeddings=embeddings)

    # bertopic modeling 결과 outlier(-1)만 나올 경우
    if np.sum(probs) == 0:
        # CountVectorizer 진행
        X = vectorizer.fit_transform(docs)

        # l2 정규화
        X = normalize(X)

        # hdbscan 알고리즘 적용
        cluster = HDBSCAN(
            min_cluster_size=2,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # trained labels
        topics = cluster.fit_predict(X.toarray())
        topics = [t + 1 for t in topics]

    else:
        threshold = 0.6
        
        # 일정 확률이 threshold보다 낮다면 outlier로 만들기
        while True :
            threshold_topics = [
                        topic if prob > threshold else -1 for topic, prob in zip(topics, probs)
                    ]

            # 만약 전부 outlier(-1)라면 threshold 내리기
            if sum(threshold_topics) == len(topics) * -1 :
                threshold -= 0.05
            else : break
        
        topics = threshold_topics 

    end = time.time()

    print(f"{end - start:.5f} sec")

    new_topics = pd.Series(topics, name="topic")
    df = df.reset_index(drop=True)
    bertopic_df = pd.concat([df, new_topics], axis=1)

    output_df = screened_articles(bertopic_df, threshold=0.3)
    output_df = output_df.reset_index(drop=True)
    return output_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="bertopic_config")
    args, _ = parser.parse_known_args()

    # input_df = pd.read_pickle("./윤석열_20221201_20221203_crwal_news_context.pkl")
    # input_df = pd.read_pickle("./윤석열_20221201_20221215_crwal_news_context.pkl")
    input_df = pd.read_pickle("./삼성전자_20221201_20221203_crwal_news_context.pkl")
    
    print(input_df)

    print("=" * 100)
    output_df = bertopic_modeling(input_df)
    output_df.to_pickle("after_bertopic.pkl")
    print(output_df)
