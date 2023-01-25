from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import sent_tokenize
import string
from sklearn.cluster import KMeans
import time
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
import random
from itertools import combinations
#nltk.download('wordnet')

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

device = torch.device('cuda:0' if torch.cuda.is_available() else 0)
model.to(device)

def make_summarization(sentences, tokenizer, model): # 문자열 생성 요약
    #print(sentences)
    if(len(sentences) < 4): return [max(sentences, key=lambda x:len(x))]
    split = []
    for i in range(len(sentences)//8):
        split.append(sentences[:8])
        sentences = sentences[8:]

    for i in range(len(split)):
        if(len(sentences) == 0): break
        split[i].append(sentences.pop())
    
    if(len(sentences) != 0): split.append(sentences)
    
    split_sum = []
    for sentences in split:
        text = '\n'.join(sentences)
        start = time.time()
        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
        
        #model.generate(torch.tensor([input_ids[:1020]]).to(device), num_beams=3, max_length=256, eos_token_id=1)
        summary_ids = model.generate(torch.tensor([input_ids]).to(device),  num_beams=4,  max_length=256, min_length=10,  eos_token_id=1)
        ##print(f"{time.time()-start:.4f} sec")
        sum_result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        sum_result = sent_tokenize(sum_result)[0]
        sum_result = delete_repeat_str(sum_result)
        ##print(sum_result)
        split_sum.append(sum_result)
        ##print(len(split), len(split_sum))
        ##print('-----------------------------------------------')
    
    if(len(split_sum)==1):
        return [split_sum[0]]
      
    return split_sum


def summarize_topic(document_df, topic_num, tokenizer, model): # df와 topic num을 넣으면 해당 topic num을 요약
    sentences = []
    numbers = []
    for i,t in enumerate(document_df[document_df['cluster_label'] == topic_num]['opinion_text']):
        sentences.append(eval(t)[1])
        numbers.append(eval(t)[0])
    result = make_summarization(sentences, tokenizer, model)
    avg = sum(numbers)/len(numbers)
    return (avg, result)


def summarize_first_sentences(processed_sentences, tokenizer, model): # 문쟈열을 k-means로 토픽 별 분류(첫 문장)
    clusternum = 1
    document_df = get_clustered_df(processed_sentences, clusternum)
    sum_result = []
    for c in range(clusternum):
        temp = get_topic_sentences(document_df, c)
        summ = summarize_topic(document_df, c, tokenizer, model)
        sum_result.append(summ)
        ##print('===================================================')
    
    first_result = (sum_result[0][0], [min(sum_result[0][1], key=lambda x:len(x))])
    ##print(first_result)
    return [first_result]
    

def summarize_topk_sentences(processed_sentences, tokenizer, model): # 문쟈열을 k-means로 토픽 별 분류
    clusternum = max(len(processed_sentences)//7, 1)
    document_df = get_clustered_df(processed_sentences, clusternum)
    sum_result = []
    for c in range(clusternum):
        temp = get_topic_sentences(document_df, c)
        ##print('---------------------------------------------------')
        summ = summarize_topic(document_df, c, tokenizer, model)
        sum_result.append(summ)
        ##print(summ)
        ##print('***************************************************')
        
    return sorted(sum_result, key= lambda x: x[0])


def get_clustered_df(sentences, clusternum):
    ##print(clusternum)
    
    document_df = pd.DataFrame()
    document_df['opinion_text'] = [str(t) for t in sentences]
    
    if(len(sentences) < 2):
        document_df['cluster_label'] = 0
        ##print('len document df', len(document_df))
        return document_df.sort_values(by=['cluster_label'])
    
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    lemmar = WordNetLemmatizer()
    
    # 토큰화한 각 단어들의 원형들을 리스트로 담아서 반환
    def LemTokens(tokens):
        return [lemmar.lemmatize(token) for token in tokens]

    # 텍스트를 Input으로 넣어서 토큰화시키고 토큰화된 단어들의 원형들을 리스트로 담아 반환
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize,
                            ngram_range=(1,2),
                            min_df=0.05, max_df=0.85)


    ftr_vect = tfidf_vect.fit_transform(document_df['opinion_text'])
    kmeans = KMeans(n_clusters=clusternum, max_iter=10000, random_state=42)
    cluster_label = kmeans.fit_predict(ftr_vect)
    
    # 군집화한 레이블값들을 document_df 에 추가하기
    document_df['cluster_label'] = cluster_label
    return document_df.sort_values(by=['cluster_label'])
    

def get_topic_sentences(df, clabel):
    lst = []
    for i,t in enumerate(df[df['cluster_label'] == clabel]['opinion_text']):
        ##print(i, t)
        lst.append(t)
    return lst 


def delete_similar(input_sentences):
    if(len(input_sentences) < 2):
        return input_sentences
    sorted_sentences = sorted(input_sentences, key=lambda x:x[1][::-1])
    prev_len = len(sorted_sentences)
    
    for i in range(5):
        prev = sorted_sentences[0]
        processed_sentences = []
        for j,sentence in enumerate(sorted_sentences[1:]):
            s1 = set(prev[1].split())
            s2 = set(sentence[1].split())
            actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
            if(actual_jaccard < 0.5): # if not similar
                processed_sentences.append(prev)
                prev = sentence
            #else:
                ##print(prev)
                ##print(sentence)
                ##print(actual_jaccard)
                ##print('-------------------------------------------')
                
        s1 = set(prev[1].split())
        
        if(len(processed_sentences) == 0):
            processed_sentences.append(prev)
            return processed_sentences
        
        s2 = set(processed_sentences[-1][1].split())
        actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
        if(actual_jaccard < 0.5): # if not similar
            processed_sentences.append(prev)
        
        sorted_sentences = sorted(processed_sentences, key=lambda x:x[1])
        
        if(prev_len == len(sorted_sentences)):
            break
        prev_len = len(sorted_sentences)
        
    return sorted_sentences


def get_first_topk_sentences(df, topk_percent):
    first_sentences = []
    topk_sentences = []
    for a,b in zip(df['context'], df['topk']):
        context = eval(str(a))
        topk = eval(str(b))
        k = int(len(topk)*(topk_percent/100))
        topk = topk[:k]
        
        first = []
        for item in topk:
            if(item[0] == 0): 
                first.append(item)
                
        if(len(first) == 0):
            first_sentences += [(0, context[0])]
            topk_sentences += topk
        else:
            first_sentences += first
            topk.remove(first[0])
            topk_sentences += topk
                
    ##print('before delete similar:', len(first_sentences), len(topk_sentences))
    first_sentences = delete_similar(first_sentences)
    topk_sentences = delete_similar(topk_sentences)
    ##print('after delete similar:', len(first_sentences), len(topk_sentences))
    return first_sentences, topk_sentences


def get_additional_topk_sentences(df, prev_topk_percent, next_topk_percent):
    topk_sentences = []
    next_topk_percent = prev_topk_percent + next_topk_percent
    for a,b in zip(df['context'], df['topk']):
        context = eval(str(a))
        topk = eval(str(b))
        pk = int(len(topk)*(prev_topk_percent/100))
        k = int(len(topk)*(next_topk_percent/100))
        topk = topk[pk:k]
        topk_sentences += topk
                
    ##print('before delete similar:', len(topk_sentences))
    if(len(topk_sentences) == 0):
        return topk_sentences
    
    topk_sentences = delete_similar(topk_sentences)
    ##print('after delete similar:', len(topk_sentences))
    return topk_sentences    


def get_topk_sentences(k, user_input, model, tokenizer):
    bot_input_ids = News_to_input(user_input, openapi_key)
    
    chat_history_ids = summary(args, bot_input_ids, -1, '', None, model)
    pred_lst = list(chat_history_ids[0][:k])
    final_text = []
    for i,a in enumerate(user_input.split('.')):
        if i in pred_lst:
            final_text.append(a+'. ')
    return final_text
    


def delete_repeat(user_text):
    text = user_text.split()
    x = len(text)
    comb = list(combinations(range(x), 2))
    sorted_comb = sorted(comb, key=lambda x: x[1]-x[0], reverse = True)
    for item in sorted_comb:
        start, end = item
        if(end-start <= len(sorted_comb)/2 and end-start>2):
            find_str = ' '.join(text[start:end])
            rest_str = ' '.join(text[end:])
            idx = rest_str.find(find_str)
            if idx != -1:
                ##print('deleted :', idx, '|', find_str)
                ##print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
                ret = ' '.join(text[:end]) + ' ' + rest_str[:idx] + rest_str[idx+len(find_str)+1:]
                ##print(user_text)
                ##print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
                ##print(ret)
                ##print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                return ret
    return user_text


def delete_repeat_str(user_text):
    prev = user_text
    new = delete_repeat(prev)
    while(new != prev):
        prev = new
        new = delete_repeat_str(prev)        
    return new


def calc_len(sum_result2):
    ret = 0
    for s in sum_result2:
        tmp = 0
        for c in s[1]:
            tmp += len(c)
        ret += tmp
    return ret


def make_summary_paragraph(topic_context_df):
    start = time.time()
    first_sentences, topk_sentences = get_first_topk_sentences(topic_context_df, 15)

    sum_result1 = summarize_first_sentences(first_sentences, tokenizer, model)
    sum_result2 = sum_result1 + summarize_topk_sentences(topk_sentences, tokenizer, model)

    prev_topk = 15

    threshold = min(570, len(topic_context_df) * 120)
    while(calc_len(sum_result2) < threshold): # 너무 짧을 때
        ##print(f'.................{calc_len(sum_result2)}<{threshold} get additional topk {prev_topk} ...............')
        topk_sentences2 = get_additional_topk_sentences(topic_context_df, prev_topk, 7)
        if(len(topk_sentences2) > 0):
            sum_result3 = summarize_topk_sentences(topk_sentences2, tokenizer, model)
            sum_result2 = sorted(sum_result3 + sum_result2, key=lambda x:x[0])
        prev_topk += 7

    final_result = ''
    for v in sum_result2:
        paragraph = [s.strip() for s in v[1]]
        new_paragraph = [s if('.' in s[-2: ]) else s+'. ' for s in paragraph]
        final_result += ' '.join(new_paragraph)+'\n\n'
    return final_result
    '''
    ##print('======================= Final result =========================')
    ##print(final_result)
    ##print(f"{time.time()-start:.4f} sec")
    ##print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    if not os.path.exists(f'./summary/{pathname}/'):
        os.makedirs(f'./summary/{pathname}/')
    f = open(f'./summary/{pathname}/{name}_summary_{i}.txt', 'w')
    f.write(final_result)
    f.close()
    '''

if __name__=="__main__":
    news_df = pd.read_pickle("para_extract_summary.pkl")
    final_result = make_summary_topic(news_df,0)
    ##print(final_result)