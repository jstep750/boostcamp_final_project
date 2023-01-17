import pandas as pd
import numpy as np
import re
#import kss
from soynlp.normalizer import *
from collections import OrderedDict
#from pykospacing import Spacing
from hanspell import spell_checker
#from konlpy.tag import Mecab

class Preprocess():

    def __init__(self):
        
        #self.spacing = Spacing()
        self.excluded_words = ["이하 뉴스1", "이 줄은 실제 뉴스"]
        #self.mecab = Mecab()

    def _split_context(self,texts):

        context = texts.split('\n')
        # sents = []

        # for sent in context:
        #     sent = sent.strip()
        #     if sent:
        #         splited_sent = kss.split_sentences(sent)
        #         sents.extend(splited_sent)

        return context
    
    def _remove_html(self, texts):
        """
        HTML 태그를 제거합니다.
        ``<p>안녕하세요 ㅎㅎ </p>`` -> ``안녕하세요 ㅎㅎ ``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text
    
    def _remove_email(self, texts):
        """
        이메일을 제거합니다.
        ``홍길동 abc@gmail.com 연락주세요!`` -> ``홍길동  연락주세요!``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _remove_hashtag(self, texts):
        """
        해쉬태그(#)를 제거합니다.
        ``대박! #맛집 #JMT`` -> ``대박!  ``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"#\S+", "", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _remove_user_mention(self, texts):
        """
        유저에 대한 멘션(@) 태그를 제거합니다.
        ``@홍길동 감사합니다!`` -> `` 감사합니다!``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"@\w+", "", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _remove_url(self, texts):
        """
        URL을 제거합니다.
        ``주소: www.naver.com`` -> ``주소: ``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text).strip()
            text = re.sub(r"pic\.(\w+\.)+\S*", "", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _remove_bad_char(self, texts):
        """
        문제를 일으킬 수 있는 문자들을 제거합니다.
        """
        bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
        preprcessed_text = []
        for text in texts:
            for bad_char in bad_chars:
                text = text.replace(bad_char, bad_chars[bad_char])
            text = re.sub(r"[\+á?\xc3\xa1]", "", text)
            if text:
                preprcessed_text.append(text)
        return preprcessed_text

    def _remove_press(self, texts):
        """
        언론 정보를 제거합니다.
        ``홍길동 기자 (연합뉴스)`` -> ````
        ``(이스탄불=연합뉴스) 하채림 특파원 -> ````
        """
        re_patterns = [
            r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
            r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장) ",  # 이름 + 기자
            r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",  # (... 연합뉴스) ..
            r"\(\s+\)",  # (  )
            r"\(=\s+\)",  # (=  )
            r"\(\s+=\)",  # (  =)
        ]

        preprocessed_text = []
        for text in texts:
            for re_pattern in re_patterns:
                text = re.sub(re_pattern, "", text).strip()
            if text:
                preprocessed_text.append(text)    
        return preprocessed_text

    def _remove_copyright(self, texts):
        """
        뉴스 내 포함된 저작권 관련 텍스트를 제거합니다.
        ``(사진=저작권자(c) 연합뉴스, 무단 전재-재배포 금지)`` -> ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` TODO 수정할 것
        """
        re_patterns = [
            r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
            r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
        ]
        preprocessed_text = []
        for text in texts:
            for re_pattern in re_patterns:
                text = re.sub(re_pattern, "", text).strip()
            if text:
                preprocessed_text.append(text)    
        return preprocessed_text

    def _remove_photo_info(self, texts):
        """
        뉴스 내 포함된 이미지에 대한 label을 제거합니다.
        ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` -> ````
        ``(출처=청주시)`` -> ````
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ", "", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _remove_useless_breacket(self, texts):
        """
        위키피디아 전처리를 위한 함수입니다.
        괄호 내부에 의미가 없는 정보를 제거합니다.
        아무런 정보를 포함하고 있지 않다면, 괄호를 통채로 제거합니다.
        ``수학(,)`` -> ``수학``
        ``수학(數學,) -> ``수학(數學)``
        """
        bracket_pattern = re.compile(r"\((.*?)\)")
        preprocessed_text = []
        for text in texts:
            modi_text = ""
            text = text.replace("()", "")  # 수학() -> 수학
            brackets = bracket_pattern.search(text)
            if not brackets:
                if text:
                    preprocessed_text.append(text)
                    continue
            replace_brackets = {}
            # key: 원본 문장에서 고쳐야하는 index, value: 고쳐져야 하는 값
            # e.g. {'2,8': '(數學)','34,37': ''}
            while brackets:
                index_key = str(brackets.start()) + "," + str(brackets.end())
                bracket = text[brackets.start() + 1 : brackets.end() - 1]
                infos = bracket.split(",")
                modi_infos = []
                for info in infos:
                    info = info.strip()
                    if len(info) > 0:
                        modi_infos.append(info)
                if len(modi_infos) > 0:
                    replace_brackets[index_key] = "(" + ", ".join(modi_infos) + ")"
                else:
                    replace_brackets[index_key] = ""
                brackets = bracket_pattern.search(text, brackets.start() + 1)
            end_index = 0
            for index_key in replace_brackets.keys():
                start_index = int(index_key.split(",")[0])
                modi_text += text[end_index:start_index]
                modi_text += replace_brackets[index_key]
                end_index = int(index_key.split(",")[1])
            modi_text += text[end_index:]
            modi_text = modi_text.strip()
            if modi_text:
                preprocessed_text.append(modi_text)
        return preprocessed_text

    def _remove_repeat_char(self, texts):
        preprocessed_text = []
        for text in texts:
            text = repeat_normalize(text, num_repeats=2).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _clean_punc(self, texts):
        punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", \
            "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', \
                'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

        preprocessed_text = []
        for text in texts:
            for p in punct_mapping:
                text = text.replace(p, punct_mapping[p])
            text = text.strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text

    def _remove_repeated_spacing(self, texts):
        """
        두 개 이상의 연속된 공백을 하나로 치환합니다.
        ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
        """
        preprocessed_text = []
        for text in texts:
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                preprocessed_text.append(text)
        return preprocessed_text
        
    def _remove_dup_sent(self, texts):
        """
        중복된 문장을 제거합니다.
        """
        texts = list(OrderedDict.fromkeys(texts))
        return texts

    # def _spacing_sent(self, texts):
    #     """
    #     띄어쓰기를 보정합니다.
    #     """
    #     preprocessed_text = []
    #     for text in texts:
    #         text = self.spacing(text)
    #         if text:
    #             preprocessed_text.append(text)
    #     return preprocessed_text

    def _spell_check_sent(self, texts):
        """
        맞춤법을 보정합니다.
        """
        preprocessed_text = []
        for text in texts:
            try:
                spelled_sent = spell_checker.check(text)
                checked_sent = spelled_sent.checked 
                if checked_sent:
                    preprocessed_text.append(checked_sent)
            except:
                preprocessed_text.append(text)
        return preprocessed_text

    
    def _excluded_word_filter(self, texts):
        """
        특정 단어를 포함하는 문장 필터링
        """
        preprocessed_text = []
        for text in texts:
            include_flag = False
            for word in self.excluded_words:
                if word in text:
                    include_flag = True
                    break
            if not include_flag:
                preprocessed_text.append(text)
        return preprocessed_text


    # def _morph_filter(self, texts):
    #     """
    #     명사(NN), 동사(V), 형용사(J)의 포함 여부에 따라 문장 필터링
    #     """
    #     NN_TAGS = ["NNG", "NNP", "NNB", "NP"]
    #     V_TAGS = ["VV", "VA", "VX", "VCP", "VCN", "XSN", "XSA", "XSV"]
    #     J_TAGS = ["JKS", "J", "JO", "JK", "JKC", "JKG", "JKB", "JKV", "JKQ", "JX", "JC", "JKI", "JKO", "JKM", "ETM"]

    #     preprocessed_text = []
    #     for text in texts:
    #         morphs = self.mecab.pos(text, join=False)

    #         nn_flag = False
    #         v_flag = False
    #         j_flag = False
    #         for morph in morphs:
    #             pos_tags = morph[1].split("+")
    #             for pos_tag in pos_tags:
    #                 if not nn_flag and pos_tag in NN_TAGS:
    #                     nn_flag = True
    #                 if not v_flag and pos_tag in V_TAGS:
    #                     v_flag = True
    #                 if not j_flag and pos_tag in J_TAGS:
    #                     j_flag = True
    #             if nn_flag and v_flag and j_flag:
    #                 preprocessed_text.append(text)
    #                 break
    #     return preprocessed_text
    
    def _remove_stopwords(self, sents):
        #  큰 의미가 없는 불용어 정의
        #  사용자가 자체적으로 stopwords를 정희하면 될 듯
        stopwords = ['소취요', '-', '조드윅', '포스터', '앓는', '서린']
        preprocessed_text = []
        for sent in sents:
            sent = [w for w in sent.split(' ') if w not in stopwords]# 불용어 제거
            preprocessed_text.append(' '.join(sent))
        return preprocessed_text

    def __call__(self, text):

        if text != None:

            context = self._split_context(text)
            context = self._remove_html(context)
            context = self._remove_email(context)
            context = self._remove_hashtag(context)
            context = self._remove_user_mention(context)
            context = self._remove_url(context)
            context = self._remove_bad_char(context)
            context = self._remove_press(context)
            context = self._remove_copyright(context)
            context = self._remove_photo_info(context)
            context = self._remove_useless_breacket(context)
            context = self._remove_repeat_char(context)
            context = self._clean_punc(context)
            context = self._remove_repeated_spacing(context)
            # context = self._remove_dup_sent(context)
            # context = self._spacing_sent(context)
            # context = self._spell_check_sent(context)
            # context = self._excluded_word_filter(context)
            # context = self._remove_stopwords(context)
            
            return ' '.join(context)
    
        else:
            return None