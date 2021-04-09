import torch
from torch import nn
from models.transformer import TransformerEncoder
from models.crf import CRF


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def init_model_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TENER(BasicModel):
    def __init__(self, config):
        super().__init__()

        self.embedding = torch.nn.Embedding(len(config.tokenizer.vocab), config.emb_size)
        self.in_fc = nn.Linear(config.emb_size, config.d_model)
        self.transformer = TransformerEncoder(config)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.out_fc = nn.Linear(config.d_model, len(config.label2id))
        self.crf = CRF(num_tags=len(config.label2id), batch_first=True)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, input_mask, labels=None, input_lens=None):
        embs = self.embedding(input_ids)
        embs = self.in_fc(embs)  # [bn, seq_len, d_model]
        embs = self.transformer(embs, input_mask)
        embs = self.fc_dropout(embs)
        logits = self.out_fc(embs)
        return logits
