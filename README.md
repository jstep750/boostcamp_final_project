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

