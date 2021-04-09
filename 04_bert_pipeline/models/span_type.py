import torch
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


class EntityType(Basic_model.BasicModel):
    def __init__(self):
        super(EntityType, self).__init__()
        self.type_fc = nn.Linear(768*2, 10)
        self.apply(self.init_model_weights)

    def gather_info(self, input_tensor, positions):
        batch_size, seq_len, hidden_size = input_tensor.size()
        flat_offsets = torch.linspace(0, batch_size-1, steps=batch_size, device=input_tensor.device).long().view(-1, 1)*seq_len
        flat_positions = positions.long() + flat_offsets
        flat_positions = flat_positions.view(-1)
        flat_seq_tensor = input_tensor.contiguous().view(batch_size*seq_len, hidden_size)
        output_tensor = torch.index_select(flat_seq_tensor, 0, flat_positions).view(batch_size, -1, hidden_size)
        return output_tensor

    def forward(self, encoder_output, s_pos, e_pos):
        entity_start_emb = self.gather_info(encoder_output, s_pos)
        entity_end_emb = self.gather_info(encoder_output, e_pos)
        entity_emb = torch.cat([entity_start_emb, entity_end_emb], dim=-1)
        entity_type_logits = self.type_fc(entity_emb)
        entity_type_probs = F.softmax(entity_type_logits, dim=-1)
        return entity_type_logits, entity_type_probs
