import re


def remove_html_entity(texts):
    """
    HTML에서 쓰이는 entity 문자를 제거합니다.
    """
    preprcessed_text = []
    for text in texts :
        text = text.replace('&quot;', '')
        text = text.replace('&apos;', '')
        preprcessed_text.append(text)

    return preprcessed_text

def remove_markdown(texts):
    """
    markdown 문자들을 제거합니다.
    """
    preprcessed_text = []
    for text in texts :
        text = text.replace('<b>', '')
        text = text.replace('</b>', '')
        preprcessed_text.append(text)
        
    return preprcessed_text

def remove_bad_char(texts):
    """
    문제를 일으킬 수 있는 문자들을 제거합니다.
    """
    bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
    preprcessed_text = []
    for text in texts:
        for bad_char in bad_chars:
            text = text.replace(bad_char, bad_chars[bad_char])
        text = re.sub(r"[\+á?\xc3\xa1]", "", text)
        preprcessed_text.append(text)
    
    return preprcessed_text

def remove_repeated_spacing(texts):
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