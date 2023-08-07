import openai
import pandas as pd

openai.api_key = "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"

engine = "gpt-3.5-turbo"
temperature = 0.7
max_tokens = 1000
stop_sequence = "\n"

def summarize_and_simplify(text):

    prompt = f"Summarize this legal provision into a shorter sentence and explain it in simple words for better understanding: \"{text}\""
    
    response = openai.ChatCompletion.create(
        engine=engine,
        messages=[{"role": "user", "content": f"Summarize this legal provision into a shorter sentence and explain it in simple words for better understanding in korean: \"{text}\""}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequence
    )


    summary = response.choices[0].text.strip()
    return summary

input_file_path = 'pre_test.csv'
df = pd.read_csv(input_file_path)
df['Summary_and_simplify'] = df.iloc[:, 0].apply(summarize_and_simplify)

output_file_path = 'api_pretest.csv'
df.to_csv(output_file_path, index=False)
