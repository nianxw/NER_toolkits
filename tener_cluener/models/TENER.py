import torch
from torch import nn
from models.transformer import TransformerEncoder
from models.crf import CRF


class NER_model(nn.Module):
    def __init__(self, ner_processor, config):
        super().__init__()

        vocab_size = len(ner_processor.vocab)
        num_labels = len(ner_processor.idx2label)
        self.embedding = torch.nn.Embedding(vocab_size, config.emb_size)
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)
        self.embed_size = config.emb_size
        self.in_fc = nn.Linear(config.emb_size, config.d_model)
        self.transformer = TransformerEncoder(config)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.out_fc = nn.Linear(config.d_model, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, input_mask, labels=None, input_lens=None):
        embs = self.embedding(input_ids)
        embs = self.in_fc(embs)  # [bn, seq_len, d_model]
        embs = self.transformer(embs, input_mask)
        embs = self.fc_dropout(embs)
        logits = self.out_fc(embs)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=input_mask)
            outputs = (-1*loss,)+outputs
        return outputs  # (loss), scores
