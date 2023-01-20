import re
import time
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from konlpy.tag import Mecab

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP

import argparse
from omegaconf import OmegaConf

import os
import sys
from pathlib import Path
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent, "")
sys.path.append(ASSETS_DIR_PATH)
from bertopic_preprocessing import *

class CustomTokenizer:
    def __init__(self, tagger, stopwords):
        self.tagger = tagger
        self.stopwords = stopwords

    def __call__(self, sent):
        # sent = sent[:1000000] # if Error?
        word_tokens = self.tagger.morphs(sent)
        result = [
            word for word in word_tokens if len(word) > 1 and word not in self.stopwords
        ]
        return result


def bertopic_modeling(cfg ,df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    file_cfg = cfg.file
    model_cfg = cfg.model 

    # 중복 제거
    drop_df = remove_duplicate(df)

    # title 과 description 합치기
    titles = drop_df["title"].to_list()
    descriptions = drop_df["description"].to_list()
    title_description_concat = [
        title + " " + des for title, des in zip(titles, descriptions)
    ]

    preprocessed_docs = execute_preprocessing(title_description_concat)
    file_path = os.path.join(Path(__file__).parent, file_cfg.stopwords_path)
    f = open(file_path, "r")
    ko_stop_words = [text.rstrip() for text in f.readlines()]

    custom_tokenizer = CustomTokenizer(Mecab(), ko_stop_words)
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

    model = BERTopic(
        embedding_model = model_cfg.model_name,
        vectorizer_model = vectorizer,
        top_n_words = model_cfg.top_n_words,
        # nr_topics = model_cfg.nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    start = time.time()
    # topics, probs = model.fit_transform(preprocessed_docs)
    model.fit_transform(preprocessed_docs)
    end = time.time()

    print(f"{end - start:.5f} sec")

    src_df = model.get_document_info(preprocessed_docs)
    bertopic_df = pd.concat([drop_df, src_df["Topic"], src_df["Top_n_words"]], axis=1)


    return bertopic_df


def remove_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    drop_df = df.drop_duplicates("title")
    drop_df = drop_df.drop_duplicates("originallink")
    drop_df = drop_df.drop_duplicates("link")
    drop_df = drop_df.drop_duplicates("description")
    drop_df = drop_df.reset_index(drop=True)
    return drop_df


def execute_preprocessing(texts: List[str]) -> List[str]:
    """_summary_

    Args:
        texts (List[str]): _description_

    Returns:
        List[str]: _description_
    """
    preprocessed_docs = remove_html_entity(texts)
    preprocessed_docs = remove_markdown(preprocessed_docs)
    preprocessed_docs = remove_bad_char(preprocessed_docs)
    preprocessed_docs = remove_repeated_spacing(preprocessed_docs)
    return preprocessed_docs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="bertopic_config")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./{args.config}.yaml")

    input_df = pd.read_csv("./crawl_result(삼성전자).csv")
    print(input_df)

    print("=" * 100)
    output_df = bertopic_modeling(cfg, input_df)
    output_df.to_csv("result.csv", index=False)
    print(cfg)
    print(output_df)
