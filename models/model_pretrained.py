# See https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT for more infos.

import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def build_biobert_model():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.pooler = Identity()
    return model, tokenizer

