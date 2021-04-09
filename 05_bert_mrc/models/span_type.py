import torch.nn as nn
import torch.nn.functional as F
from models import Basic_model


class EntitySpan(Basic_model.BasicModel):
    def __init__(self, config):
        super(EntitySpan, self).__init__()
        self.start_position_fc = nn.Linear(config.hidden_size, 1)
        self.end_position_fc = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_model_weights)

    def forward(self, encoder_output):
        s_logits = F.sigmoid(self.start_position_fc(encoder_output))
        e_logits = F.sigmoid(self.end_position_fc(encoder_output))
        return s_logits, e_logits
