import os
import pickle as pickle
from pydoc import locate
from datetime import datetime

import pandas as pd

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import data_loaders.data_loader as dataloader
from data_loaders.data_loader import MyDataCollatorWithPadding
import utils.util as utils


def inference(conf):
    # 실행 시간을 기록합니다.
    now = datetime.now()
    inference_start_time = now.strftime("%d-%H-%M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = conf.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = MyDataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    model.parameters
    model.to(device)
    model.eval()

    ## load predict datset
    RE_predict_dataset = dataloader.load_predict_dataset(tokenizer, conf.path.predict_path, conf)
    RE_test_dataset = dataloader.load_dataset(tokenizer, conf.path.test_path, conf)

    # init trainer
    test_args = TrainingArguments(output_dir="./prediction", do_train=False, do_predict=True, per_device_eval_batch_size=16, dataloader_drop_last=False)
    trainer = Trainer(model=model, args=test_args, compute_metrics=utils.compute_metrics, data_collator=data_collator)

    # Test 점수 확인
    predict_dev = True  # dev set에 대한 prediction 결과값 구하기 (output분석)
    predict_submit = False  # dev set은 evaluation만 하고 submit할 결과값 구하기
    if predict_dev:
        outputs = trainer.predict(RE_test_dataset)
        logits = torch.FloatTensor(outputs.predictions)
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        result = torch.argmax(logits, axis=-1).detach().cpu().numpy()

        pred_answer = result.tolist()
        pred_answer = utils.num_to_label(pred_answer)
        output_prob = prob.tolist()

        output = pd.read_csv("./dataset/finance_data.csv")
        output["pred_label"] = pred_answer
        output["probs"] = output_prob

        output.to_csv(os.path.join(path, f"dev_submission_{inference_start_time}.csv"), index=False)
        output.to_csv(f"./prediction/dev_submission_{inference_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    if predict_submit:
        metrics = trainer.evaluate(RE_test_dataset)
        print("Training is complete!")
        print("==================== Test metric score ====================")
        print("eval loss: ", metrics["eval_loss"])
        print("eval auprc: ", metrics["eval_auprc"])
        print("eval micro f1 score: ", metrics["eval_micro f1 score"])

        outputs = trainer.predict(RE_predict_dataset)
        logits = torch.FloatTensor(outputs.predictions)
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        result = torch.argmax(logits, axis=-1).detach().cpu().numpy()

        pred_answer = result.tolist()
        pred_answer = utils.num_to_label(pred_answer)
        output_prob = prob.tolist()

        output = pd.read_csv("./prediction/sample_submission.csv")
        output["pred_label"] = pred_answer
        output["probs"] = output_prob

        output.to_csv(os.path.join(path, f"submission_{inference_start_time}.csv"), index=False)
        output.to_csv(f"./prediction/submission_{inference_start_time}.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("==================== Inference finish! ====================")