import torch
from torch import nn
from typing import Optional
from .qformer import BertLMHeadModel, BertConfig
# from header import *

class TextFcLayer(nn.Module):
    """Layers used in mapping text embeddings to visual outputs."""

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2, cross_attention_freq=1):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = num_hidden_layers
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1,
                 mode: str = 'linear',
                 freeze_qformer=False):
        """
        :param mode: ['linear', 'transformer', 'qformer']
        :param freeze_qformer: whether freeze the weights of qformer
        """
        super().__init__()

        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.mode = mode
        self.out_dim = out_dim

        if mode == 'linear':
            self.model = nn.Linear(in_dim, out_dim) 
            self.pooling = nn.AdaptiveAvgPool1d(num_output_tokens)
        elif mode == 'linear-binary':
            self.l1 = nn.Linear(in_dim, out_dim)  # [1,32,1536] -> [1,32,2048]
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.2)
            self.l2 = nn.Linear(num_input_tokens, num_output_tokens)  # [1,32,2048] -> [1,256,2048]
        elif mode == 'transformer':
            hidden_dim = 1536
            self.in_fc = nn.Linear(in_dim, hidden_dim)
            self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                    d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                    dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
            self.out_fc = nn.Linear(hidden_dim, out_dim)
            self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
            self.query_embs.data.normal_(mean=0.0, std=0.0)
        elif mode == 'qformer':
            print('Loading Q-Former')
            hidden_dim = 768
            self.fc = nn.Linear(in_dim, hidden_dim)
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_output_tokens, hidden_dim
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.model = nn.Linear(hidden_dim, out_dim)
            print('Loading Q-Former Done')
        else:
            raise NotImplementedError(mode)

    def forward(self, x: torch.Tensor, input_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = None
        if self.mode == 'linear':
            #print(x.shape)
            x_mapped = self.model(x)  # Linear: [num_input_tokens, feature_dim] -> [num_input_tokens, output_feature_dim]
            if self.num_input_tokens != self.num_output_tokens:
                x_pooled = self.pooling(x_mapped.transpose(1, 2)).transpose(1, 2)  # Pooling: [num_input_tokens, output_feature_dim] -> [num_output_tokens, output_feature_dim]
            else:
                x_pooled = x_mapped
            outputs = x_pooled
        elif self.mode == 'linear-binary':
            x_l1 = self.l1(x)  # [1,32,1536] -> [1,32,2048]
            x_l1 = self.relu(x_l1)
            x_l1 = self.dropout(x_l1)
            x_l1_transposed = x_l1.transpose(1, 2)  # [1,32,2048] -> [1,2048,32]
            x_l2 = self.l2(x_l1_transposed)  # [1,2048,32] -> [1,2048,256]
            outputs = x_l2.transpose(1, 2)  # [1,2048,128] -> [1,256,2048]
        elif self.mode == 'transformer':       
            x = self.in_fc(x)
            batch_size = x.size(0)
            query_embs = self.query_embs.expand(batch_size, -1, -1)
            x = self.tfm(x, query_embs)
            x = self.out_fc(x)
            return x
        
        elif self.mode == 'qformer':
            x = x + input_embs
            x = self.fc(x)
            image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(x.device)
            query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
            outputs = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=x,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ).last_hidden_state
            outputs = self.model(outputs)
        else:
            raise NotImplementedError(self.mode)

        assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * self.out_dim), (outputs.shape, self.num_output_tokens)
        return outputs  # (N, T_I_V_A.txt, D)