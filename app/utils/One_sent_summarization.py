from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
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