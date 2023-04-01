import math
import numpy as np
import pandas as pd
import random
import re
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

koGPT_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("kakaobrain/kogpt",
                                                            revision='KoGPT6B-ryan1.5b-float16',
                                                            bos_token=BOS, eos_token=EOS,
                                                            unk_token="<unk>", pad_token=PAD,
                                                            mask_token=MASK,
                                                            )

tokenizer = koGPT_TOKENIZER
model = GPT2LMHeadModel.from_pretrained("../finetuned_models/kakao_gpt")


with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))