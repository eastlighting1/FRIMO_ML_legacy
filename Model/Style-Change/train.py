from transformers import pipeline
import argparse
import pandas as pd

def generate_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
  target_style_name = style_map[target_style]
  text = f"{target_style_name} 말투로 변환:{text}"
  out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
  return [x['generated_text'] for x in out]

model_name = "gogamza/kobart-base-v2"
model_path = "C:/Users/User/Desktop/model/smile_style/output"

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat_sentence',
                    type=str,
                    default=True,
                    help='ai chatting data')

parser.add_argument('--chat_style',
                    type=str,
                    default=True,
                    help='ai chatting style')

style_map = {
    'formal': '문어체',
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}

nlg_pipeline = pipeline('text2text-generation',model=model_path, tokenizer=model_name)

args = parser.parse_args()

src_text = args.chat_sentence
style = args.chat_style

print("입력 문장:", src_text)
print("수정 문장 : ", generate_text(nlg_pipeline, src_text, style, num_return_sequences=1, max_length=1000)[0])
