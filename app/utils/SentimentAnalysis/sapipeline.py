from transformers import Pipeline, pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import os
import sys
from pathlib import Path
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent, "")

#epoch10 = "step_saved_model/klue-roberta-base/27-01-53/checkpoint-2000/pytorch_model.bin"
#epoch5
#epoch7 = "best_model/klue-roberta-base/30-05-22/pytorch_model.bin"
def num_to_label(labels):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = {0 : "negative", 1 : "neutral", 2 : "positive"}
    output = []
    for label in labels:
        output.append(origin_label[label])
    return output

def piplines(text_list):
    model_path = os.path.join(ASSETS_DIR_PATH,"pytorch_model_10.bin")
    print(model_path)
    model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base",num_labels=3)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    result = []
    for prediction in predictions:
        label = (prediction==max(prediction)).nonzero().squeeze()
        result += num_to_label([int(label)])
    return result

def sentiment_analysis(df:pd.DataFrame):
    output = piplines(list(df['one_sent']))
    output = pd.DataFrame(output,columns=['sentiment'])
    output = pd.concat([df,output],axis=1)
    return output

if __name__ == "__main__":
    text = ["테크노폴리스는 컴퓨터 기술과 통신 분야에서 일하는 회사들을 유치하기 위해 10만 평방미터 이상의 면적을 단계적으로 개발할 계획이라고 성명은 밝혔다."]
    test_df = pd.read_csv("sentiment.csv")
    test_df = test_df.reset_index(drop=True)    
    output = sentiment_analysis(test_df)
    output.to_csv("sentiment.csv",index=False)
    print(output)
