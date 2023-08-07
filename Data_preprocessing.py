import pandas as pd
from konlpy.tag import Okt

okt = Okt()

def tokenize_function(text):
    tokens = okt.morphs(text)
    return tokens

def split_summary_over_1024_tokens(data, file_name):
    new_rows = []
    
    for index, row in data.iterrows():
        summary = row['summary']
        tokens = tokenize_function(summary)
        
        if len(tokens) > 1024:
            sentences = summary.split('. ')
            
            new_summary = []
            token_count = 0
            for sentence in sentences:
                sentence_tokens = tokenize_function(sentence)
                token_count += len(sentence_tokens)
                
                if token_count <= 1024:
                    new_summary.append(sentence)
                else:
                    break
                    
            new_text = '. '.join(new_summary)
            remaining_text = '. '.join(sentences[len(new_summary):])
            
            while len(remaining_text) > 0:
                remaining_tokens = tokenize_function(remaining_text)
                
                if len(remaining_tokens) > 1024:
                    sentences = remaining_text.split('. ')
                    
                    remaining_summary = []
                    token_count = 0
                    for sentence in sentences:
                        sentence_tokens = tokenize_function(sentence)
                        token_count += len(sentence_tokens)
                        
                        if token_count <= 1024:
                            remaining_summary.append(sentence)
                        else:
                            break
                            
                    chunk_text = '. '.join(remaining_summary)
                    remaining_text = '. '.join(sentences[len(remaining_summary):])
                else:
                    chunk_text = remaining_text
                    remaining_text = ''
                
                new_rows.append({'summary': chunk_text})
        else:
            new_text = tokens
        
        data.loc[index, 'summary'] = new_text

    new_data = data.append(pd.DataFrame(new_rows), ignore_index=True)
    new_data.to_csv(file_name, index=False)


data1 = pd.read_csv("plustrain.csv")
data2 = pd.read_csv("plusval.csv")
data3 = pd.read_csv("plustest.csv")
data4 = pd.read_csv("train.csv")
data5 = pd.read_csv("val.csv")
data6 = pd.read_csv("test.csv")

split_summary_over_1024_tokens(data1, "new_plustrain.csv")
split_summary_over_1024_tokens(data2, "new_plusval.csv")
split_summary_over_1024_tokens(data3, "new_plustest.csv")
split_summary_over_1024_tokens(data4, "new_train.csv")
split_summary_over_1024_tokens(data5, "new_val.csv")
split_summary_over_1024_tokens(data6, "new_test.csv")
