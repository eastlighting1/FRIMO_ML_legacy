import torch
import argparse
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat_data',
                    type=str,
                    default=False,
                    help='user chatting data')

args = parser.parse_args()

text = args.chat_data

if text:
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)