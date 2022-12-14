from transformers import (
    AutoConfig, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer, TrainingArguments,AutoModelWithLMHead
)
from typing import Dict, List, Optional
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from IPython.display import display
from typing import Dict

df = pd.read_csv("./dataset/smilestyle_dataset.tsv", sep="\t")

row_notna_count = df.notna().sum(axis=1)
df = df[row_notna_count >= 2]
print(len(df))

model_name = "beomi/kcgpt2"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token