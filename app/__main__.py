if __name__ == "__main__":
    import uvicorn
    import nltk
    nltk.download('wordnet')
    nltk.download('punkt')
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
