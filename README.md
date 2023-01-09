## 실행 환경
aistage 서버 python version 이 3.9.13이라 여기에 맞춤

```
conda create -n final_project python=3.9.13
pip install -r requirements.txt
bash ./install.sh # hanspell은 pip에 없음
```

새로운 패키지를 설치 했을때
```
pip list --format=freeze > ./requirements.txt
```

## streamlit run
```
streamlit run main.py
```

