## 실행 환경
aistage 서버 python version 이 3.8.5 이라 여기에 맞춤(3.8.5 가 맞음)

```
conda create -n final_project
pip install -r requirements.txt
bash ./install.sh # hanspell은 pip에 없음
```

새로운 패키지를 설치 했을때
```
pip list --format=freeze > ./requirements.txt
```

## streamlit run
```
streamlit run main.py --server.port 30001
```

------
feat_front readme
# requirement
```
pip install -r app/requirements.txt
```

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
# KoBert 사용
```
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```
# 파일 구성
```
📁app
│   └── __main__.py # Backend 실행 파일
│   └── main.py #Backend
📁front
│   └── main.py #Frontend
│   └── utils.py #크롤링, 전처리, 모델실행 등
│   └── style.css # Frontend css 파일
```
