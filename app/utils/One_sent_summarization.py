from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizerFast, BartForConditionalGeneration
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

import torch
import pandas as pd
import numpy as np
import time
import sys
import requests
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocessing(text):
    text = text.replace('<b>', '')
    text = text.replace('</b>', '')
    text = text.replace('&apos;', '')
    text = text.replace('...', '')
    text = text.replace('\\u200b', '')
    text = text.replace('M&amp;A', '')
    text = text.replace('&quot;', '')
    text = text.replace('\\xa0', '')
    text = text.strip()
    return text

def summary_one_sent(topic_number:int, df: pd.DataFrame = None) -> pd.DataFrame:
    add_title_context = []
    for _, t in df.iterrows():
        #tem = [c for c in t['context'].split('.') if len(c) > 3]
        tem = t['context']
        try:
            con = t['title'] + ' ' + tem[0][1] + '. ' + tem[1][1] + '.'
        except:
            con = t['title'] + ' ' + tem[0][1] + '.'
        con = preprocessing(con)
        add_title_context.append(con)

    title_context = [preprocessing(con) for con in df.title]

    Kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    Kobart = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    Kobart.to(device)

    s = time.time()
    input_ids = Kobart_tokenizer.encode('.'.join(add_title_context))
    #print('input_ids length: ', len(input_ids))
    summary_ids = Kobart.generate(torch.tensor([input_ids[:1020]]).to(device),  num_beams=4,  max_length=512,  eos_token_id=1)
    one_sentence = Kobart_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    #print(time.time() - s)
    news_df = pd.DataFrame(data = {'topic':[topic_number],'one_sent':[one_sentence]})
    return news_df
    
if __name__ == "__main__":
    news_df = pd.read_pickle("/opt/ml/final-project-level3-nlp-05/after_bertopic.pkl")
    topic_df = pd.DataFrame()
    #토픽번호에 맞는 데이터만 가져오기
    print("total topic num : ",set(news_df['topic']))
    for topic_number in set(news_df['topic']):
        print("topic_number",topic_number)
        if topic_number == -1:
            continue
        now_news_df = news_df[news_df['topic']==topic_number]
        print("len ",len(now_news_df))
        now_topic_df = summary_one_sent(topic_number,now_news_df)
        topic_df = pd.concat([topic_df,now_topic_df])
    print(topic_df)