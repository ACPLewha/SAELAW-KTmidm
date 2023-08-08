import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

data6 = pd.read_csv("pre_test.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
from konlpy.tag import Okt
from transformers import BertForSequenceClassification, BertTokenizer
from heapq import nlargest

from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt2_model.eval()

def rephrase_gpt2(tokenizer, model, summarized_text, max_length=200):
    prompt = f"Please rephrase this legal terminology into a simple sentence that retains the original meaning but is easy for a child to understand: {summarized_text}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                            no_repeat_ngram_size=2, early_stopping=True)
    
    rephrased_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return rephrased_text


def get_kobert_model():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2)
    model.eval()
    
    return tokenizer, model

def tokenize_sentences(tokenizer, sentences):
    encoded_input = tokenizer(sentences, return_tensors='pt', padding=True,
                                truncation=True, max_length=200)
    return encoded_input

def kobert_summarize(tokenizer, model, text, top_n=3):
    okt = Okt()
    sentences = text.split(". ")
    
    sentences_with_token = okt.pos(text)
    words = [word for word, _ in sentences_with_token]
    unique_words = list(set(words))

    tokenized_sentences = tokenize_sentences(tokenizer, unique_words)

    with torch.no_grad():
        outputs = model(**tokenized_sentences).logits
    
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    relevance_scores = {w: outputs[i][1].item() for i, w in enumerate(unique_words)}

    sentence_scores = {}
    for s in sentences:
        tokenized_sentence = okt.pos(s)
        sentence_scores[s] = sum([relevance_scores[word] for word in tokenized_sentence if word in relevance_scores])

    summarized_sentences = nlargest(top_n, sentence_scores, key=sentence_scores.get)
    
    return ". ".join(sorted(summarized_sentences, key=text.find))


def process(data):
    kobert_tokenizer, kobert_model = get_kobert_model()
    rephrased_result = []
    
    for index, row in data.iterrows():
        original_text = row["summary"]
        if pd.isna(original_text): 
            rephrased_result.append(None)
        else:
            summarized_text = kobert_summarize(kobert_tokenizer, kobert_model, str(original_text))
            rephrased_text = rephrase_gpt2(gpt2_tokenizer, gpt2_model, summarized_text)
            rephrased_result.append(rephrased_text)  
        
        print("rephrased_text", rephrased_text)
        
    return rephrased_result


