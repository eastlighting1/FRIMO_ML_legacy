from transformers import pipeline
import argparse

parser = argparse.ArgumentParser(description='Style Change')

parser.add_argument('--model_path',
                    type=str,
                    default=True,
                    help='trained_model_path')

parser.add_argument('--style',
                    type=str,
                    default=True,
                    help='style')

parser.add_argument('--sentence',
                    type=str,
                    default=True,
                    help='style')

args = parser.parse_args()

model_name = "gogamza/kobart-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = args.model_path
style = args.style
src_text = args.sentence

nlg_pipeline = pipeline('text2text-generation',model=model_path, tokenizer=model_name)

def generate_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
  target_style_name = style_map[target_style]
  text = f"{target_style_name} 말투로 변환:{text}"
  out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
  return [x['generated_text'] for x in out]

print(generate_text(nlg_pipeline, src_text, style, num_return_sequences=1, max_length=1000)[0])
