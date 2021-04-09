from torch.nn import LayerNorm
import torch.nn as nn
from models.crf import CRF


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        # (N, K, 1, T), some features are masked
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def init_model_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BilstmCrf(BasicModel):
    def __init__(self,
                 args):
        super(BilstmCrf, self).__init__()
        self.embedding = nn.Embedding(len(args.tokenizer.vocab), args.embedding_size)
        self.bilstm = nn.LSTM(input_size=args.embedding_size, hidden_size=args.hidden_size,
                              batch_first=True, num_layers=2, dropout=0.1,
                              bidirectional=True)
        self.dropout = SpatialDropout(0.1)
        self.layer_norm = LayerNorm(args.hidden_size * 2)
        self.classifier = nn.Linear(args.hidden_size * 2, args.num_labels)
        self.crf = CRF(num_tags=args.num_labels, batch_first=True)
        self.apply(self.init_model_weights)

    def forward(self, input_ids, input_mask):
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)
        logits = self.classifier(seqence_output)
        return logits
