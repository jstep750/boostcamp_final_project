# 뉴스 데이터 요약 및 분석 서비스

## Table of content

* Intro : 팀 소개/ 프로젝스 소개(문제 정의) / 개발 목표
* Model/ Reasearch: 데이터셋 / 모델 / 연구 / 최종 적용 모델
* Product Serving: 아키텍쳐/ 구현/ 데모
* Result / Conclusion: 시연 영상 / 후속 개발 및 연구 / 결과 및 고찰
* Appendix: 도전적인 실험 / 레슨런 / 예상 Q&A / 팀원 개별 소개 등

## Intro

### Objective
투자를 할 때 뉴스 데이터를 잘 활용할 수 있게 돕는 웹서비스 제작

### Project intro
투자를 할 때 네이버 뉴스에 기업명을 최신순으로 검색하는 경우가 많다. 이렇게 하면 유사한 뉴스가 많이 나오고 본문에 기업 이름이 포함되어 있지만 관련성이 적은 기사까지 검색결과로 나와서 원하는 정보만 찾기 어렵다. 
따라서 기업과 관련된 뉴스만 수집하여 **중복 뉴스를 하나로** 합치고 **요약**하여 보여주는 서비스를 제공하고자 한다.

### Team member
김진호                       |  신혜진                   |  이효정                    |  이상문                    |  정지훈                    |
:-------------------------:|:------------------------:|:------------------------:|:------------------------:|:-------------------------:
<img src="./asset/kk_image.png" width=25% height=25%>    | <img src="./asset/kk_image.png" width=25% height=25%>  | <img src="./asset/kk_image.png" width=25% height=25%>  | <img src="./asset/kk_image.png" width=75% height=75%> | <img src="./asset/kk_image.png" width=25% height=25%> 
| 토픽 모델링  | 본문 요약 모델링| 프론트, 백엔드| 뉴스 데이터 수집| 한줄 요약 모델링

## Model/ Research

### dataset
Naver news crawl api 활용 <br>
bigkinds 공공 뉴스데이터 활용

### Model

### Research

## Product Serving

### Architecture

## Result

### 구현

### Demo

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

## Result / Conclusion

### 시연영상

## Appendix

### bla bla

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
