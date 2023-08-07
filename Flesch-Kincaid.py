import re
from konlpy.tag import Okt

def count_syllables(text):
    return len(re.findall(r'\S', text))

def count_words(text):
    okt = Okt()
    words = okt.morphs(text)
    return len(words)

def count_sentences(text):
    return len(re.findall(r'[.!?]\s*', text))

def calculate_flesch_kincaid(text):
    syllables = count_syllables(text)
    words = count_words(text)
    sentences = count_sentences(text)

    if words == 0 or sentences == 0:
        return 0
    
    avg_words_per_sentence = words / sentences
    avg_syllables_per_word = syllables / words

    readability = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59

    return readability

text = "안녕하세요? 오늘 날씨가 정말 좋네요. 함께 산책하러 갈까요?"

fk_score = calculate_flesch_kincaid(text)
print(f"Flesch-Kincaid Readability Score: {fk_score:.2f}")
