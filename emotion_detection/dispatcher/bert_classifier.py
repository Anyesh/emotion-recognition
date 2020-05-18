from emotion_detection.config import config
import transformers
import torch.nn as nn


class EmotionDetection(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmotionDetection, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.hidden_size = input_size
        self.output_size = output_size
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, ids, attention_mask, token_type_ids):
        output_1, output_2 = self.bert(ids, attention_mask, token_type_ids)

        bert_output = self.bert_drop(output_2)
        output = self.out(bert_output)
        return output
