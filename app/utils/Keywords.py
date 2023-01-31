from konlpy.tag import Komoran
from collections import Counter
import pandas as pd

# 형태소 분석기
kom = Komoran(userdic="./custom_dict.txt")
# 불용어 가져오기
f = open("stop_words.txt")
stopwords = f.read().splitlines()


def keywords(df):
    keywords_df = pd.DataFrame()
    topic_nums = sorted(df["topic"].unique())
    concat_text = ""
    for topic in topic_nums:
        for text in df[df["topic"] == topic]["context"]:
            concat_text += " ".join(text)
        noun = [n[0] for n in kom.pos(concat_text) if n[1] == "NNP" and len(n[0]) > 1 and n[0] not in stopwords]
        count = Counter(noun)
        freq_vocab = [n for n, _ in count.most_common(3)]
        keywords_df[topic] = freq_vocab

    return keywords_df


if __name__ == "__main__":
    news_df = pd.read_pickle("/opt/ml/final-project-level3-nlp-05/after_bertopic.pkl")
    topic_keywords_df = pd.DataFrame()
    # 토픽번호에 맞는 데이터만 가져오기
    print("total topic num : ", set(news_df["topic"]))
    topic_keywords_df = keywords(news_df)
    print(topic_keywords_df)
