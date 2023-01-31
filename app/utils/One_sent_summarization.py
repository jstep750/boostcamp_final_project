from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from collections import Counter

import torch
import time
import pandas as pd


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

class SummaryGenerater():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(device)
    
    def hardVotingCategory(self, df: pd.DataFrame) -> pd.DataFrame :
        topic_idx = df["topic"].unique().tolist()
        hard_category1 = []
        hard_category2 = []

        for i in topic_idx :
            category1_list = df[(df["topic"])== i]['category1'].tolist()
            category2_list = df[(df["topic"])== i]['category2'].tolist()

            category = [c1 + ',' + c2 for c1, c2 in zip(category1_list, category2_list)]
            max_category = Counter(category).most_common(1)[0][0]

            max_category1, max_category2 = max_category.split(",")

            hard_category1 += [max_category1] 
            hard_category2 += [max_category2]
        
        hard_category1 = pd.Series(hard_category1, name="hard_category1")
        hard_category2 = pd.Series(hard_category2, name="hard_category2")
        
        hard_category_df = pd.concat([hard_category1, hard_category2], axis=1)
        return hard_category_df

    def summary(self, df):
        summary_dict = {}
        topic_nums = sorted(df['topic'].unique())
        
        for topic_n in topic_nums:
            if topic_n == -1:
                continue
            topic_context = list(df[df['topic'] == topic_n]['concat_text'])
            # s = time.time()
            input_ids = self.tokenizer.encode('.'.join(topic_context))
            summary_ids = self.model.generate(torch.tensor([input_ids[:1020]]).to(device), num_beams=3, max_length=256, eos_token_id=1)
            summary_text = self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            summary_dict[topic_n] = summary_text
            # print("================   Topic #: ", topic_n)
            # print("================   Article #: ", len(topic_context))
            # print(topic_context)
            # print("================")
            # print(topic_n, summary_text)
            # print(f'{(time.time() - s):0.2f} sec')
        summary_df = pd.DataFrame(data={'topic': summary_dict.keys(),
                                        'one_sent': summary_dict.values()})

        hard_category_df = self.hardVotingCategory(df)

        summary_df = pd.concat([summary_df, hard_category_df], axis=1)

        return summary_df
    
    
SG = SummaryGenerater(model, tokenizer)

def summary_one_sent(df):
    topic_df = SG.summary(df)
    return topic_df
    
if __name__ == "__main__":
    news_df = pd.read_pickle("/opt/ml/final-project-level3-nlp-05/after_bertopic.pkl")
    topic_df = pd.DataFrame()
    #토픽번호에 맞는 데이터만 가져오기
    print("total topic num : ",set(news_df['topic']))
    topic_df = SG.summary(news_df)
    print(topic_df)