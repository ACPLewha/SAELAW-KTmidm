# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Pre-trained model
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

sentence1 = "안녕하세요, 오늘 날씨가 좋네요."
sentence2 = "여보세요, 날씨가 참 맑습니다."
sentence3 = "오늘 점심 메뉴는 무엇인가요?"


embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)


similarity12 = 1 - cosine(embedding1, embedding2)
similarity13 = 1 - cosine(embedding1, embedding3)

print(f"문장 1과 문장 2의 유사성: {similarity12:.3f}")
print(f"문장 1과 문장 3의 유사성: {similarity13:.3f}")
