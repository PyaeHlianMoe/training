from fastai.text import *
from transformers import PreTrainedModel


# defining our model architecture
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel, pad_idx):
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model
        self.pad_idx = pad_idx
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask=None):
        attention_mask = (input_ids != self.pad_idx).type(input_ids.type())
        token_type_ids = (attention_mask == -2).type(input_ids.type())

        pooled_output = self.transformer(input_ids,
                                         attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        logits = self.relu(pooled_output)
        logits = self.dropout(logits)
        logits = self.softmax(logits)
        return logits
