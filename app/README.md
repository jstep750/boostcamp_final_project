# Frontend 실행
final-project-level3-nlp-05 폴더 내에서 `streamlit run frontend/main.py --server.port=30001` 실행

# Backend 실행
final-project-level3-nlp-05 폴더 내에서 `python -m app` 실행

# 버토픽 gpu에서 실행하기
```
pip install cupy-cuda11x
pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
pip install cupy-cuda110
```
# KorBertSum setting
- `app/utils/KorBertSum` 에서 001_bert_morp_pytorch.zip 앞축해제
```
📁app/utils/KorBertSum
│   └──📁001_bert_morp_pytorch
```
- `app/utils/KorBertSum/bert_models/bert_classifier2` 에서 model_step_35000.zip 앞축해제
```
📁app/utils/KorBertSum/bert_models/bert_classifier2
│   └──model_step_35000.pt
```

# KoBert 사용
```
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```
# 파일 구성
```
📁app
│   └── __main__.py # Backend 서버 실행 파일
│   └── main.py #Backend 파일
│   └──📁utils #크롤링, 요약, 모델실행 등
│       └──📁BERTopic   #topic 추출
│       └──📁KorBertSum #문단요약(추출요약, 생성요약)
│       └── Bigkindscrawl.py # DB 서버로부터 크롤링 데이터 추출
│       └── One_sent_summarization.py  #한줄요약
📁front
│   └── main.py #Frontend
│   └── style.css # Frontend css 파일
```
