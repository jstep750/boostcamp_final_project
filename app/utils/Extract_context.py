# 공백 정리
import re, unicodedata
from string import whitespace, punctuation
from pykospacing import Spacing
from nltk import word_tokenize, sent_tokenize
import pandas as pd
from tqdm import tqdm
from newspaper import Article
import time
import nltk
nltk.download('punkt')

def context(x):
    try:
        article = Article(x, language='ko')
        article.download()
        article.parse()
        return article.text
    except:
        return None


def add_context_to_df(df):
    contexts = []
    for i, link in enumerate(tqdm(df['url'])):
        cont = context(link)
        
        if(cont):
            contexts.append(cont)
        else:
            contexts.append('delete this')
    df['text'] = contexts
    df = df.drop(df[df['text'] == 'delete this'].index)
    return df


def clean_byline(text):
    # byline
    pattern_email = re.compile(r'[-_0-9a-z]+@[-_0-9a-z]+(?:\.[0-9a-z]+)+', flags=re.IGNORECASE)
    pattern_url = re.compile(r'(?:https?:\/\/)?[-_0-9a-z]+(?:\.[-_0-9a-z]+)+', flags=re.IGNORECASE)
    pattern_others = re.compile(r'\.([^\.]*(?:기자|특파원|교수|작가|대표|논설|고문|주필|부문장|팀장|장관|원장|연구원|이사장|위원|실장|차장|부장|에세이|화백|사설|소장|단장|과장|기획자|큐레이터|저작권|평론가|©|©|ⓒ|\@|\/|=|▶|무단|전재|재배포|금지|\[|\]|\(\))[^\.]*)$')
    result = pattern_email.sub('', text)
    result = pattern_url.sub('', result)

    # 본문 시작 전 꺽쇠로 쌓인 바이라인 제거
    pattern_bracket = re.compile(r'^((?:\[.+\])|(?:【.+】)|(?:<.+>)|(?:◆.+◆)\s)')
    result = pattern_bracket.sub('', result).strip()
    return result


def text_filter(text): # str -> 전처리 -> 문장 배열
    text = clean_byline(text)
    exclude_pattern = re.compile(r'[^\% 0-9a-zA-Zㄱ-ㅣ가-힣.,]+')
    exclusions = exclude_pattern.findall(text)
    result = exclude_pattern.sub(' ', text).strip()
    spacing = Spacing()
    kospacing_txt = spacing(result) 
    sentences = sent_tokenize(kospacing_txt) 
    return sentences


def text_filter2(text): # 제목, description 전처리
    text = clean_byline(text)
    exclude_pattern = re.compile(r'[^\% 0-9a-zA-Zㄱ-ㅣ가-힣.,]+')
    exclusions = exclude_pattern.findall(text)
    result = exclude_pattern.sub(' ', text).strip()
    return result


def extract_context(df:pd.DataFrame) -> pd.DataFrame:
    df = add_context_to_df(df)

    start = time.time()
    text = []
    for i,context in enumerate(df['text']):
        preprocess = text_filter(context)[:60]
        if(len(preprocess) > 3):
            text.append([(i,v) for i,v in enumerate(preprocess)])
        else:
            text.append('Hello world')
            
    df['context'] = text
    df = df.drop(df[df['context'] == 'Hello world'].index)
    df = df.drop(columns=['text'])
    df['title'] = df['title'].apply(text_filter2)
    df['description'] = df['description'].apply(text_filter2)
    #df = df.drop(columns= ['Unnamed: 0.1','Unnamed: 0'])
    print(f"{time.time()-start:.4f} sec")
    return df



if __name__ == "__main__":
    df = make_df()
    df.to_csv("context_result.csv",index=False)


