import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Optional, Union
from transformers import Qwen2Config
# from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
from .modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
from transformers.cache_utils import DynamicCache

from .masks import *
# from masks import *

IGNORE_ID = -100

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        
    def forward(self, logits, target, target_subsampling_factor=1):
        """
        logits: B*T1*D
        target: B*T2
        """
        logits = logits[:, :target.shape[1], :]
        logits = logits.transpose(1, 2)
        target = target.to(torch.long)
        loss = self.criterion(logits, target)
        return loss

class LLM2TTSCodecAR(torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (namespace): argument Namespace containing options

    """

    def __init__(self, args):
        """Initialize transducer modules.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        super(LLM2TTSCodecAR, self).__init__()
        print("warning!!!!!! initializing original AR V2 model for training!!!!!")
        self.idim = args.idim
        self.odim = args.odim
        self.encoder_pre_norm_type = args.encoder_pre_norm_type
        self.encoder_drop_rate = args.encoder_drop_rate
        self.encoder_criterion = args.encoder_criterion
        self.encoder_upsample_rate = args.encoder_upsample_rate
        # self.do_cross_attention = args.do_cross_attention
        # self.cross_attention_layer_num = args.cross_attention_layer_num
        self.reporter = None

        self.vocab_size = self.odim
        self.llm_vocab_size = args.llm_vocab_size
        config = Qwen2Config(vocab_size=self.vocab_size + 4, hidden_size=args.transformer_attention_dim, 
                            intermediate_size=args.transformer_linear_units, 
                            num_hidden_layers=args.transformer_num_blocks, 
                            num_attention_heads=args.transformer_attention_heads,
                            num_key_value_heads=args.transformer_kv_heads,
                            max_position_embeddings=2048, 
                            bos_token_id=self.vocab_size + 1, 
                            eos_token_id=self.vocab_size + 2, pad_token_id=self.vocab_size + 3,
                            attention_dropout=args.transformer_dropout_rate)

        # self.in_fnn = nn.Linear(self.idim, args.transformer_attention_dim)

        self.embed_tokens = nn.Embedding(self.llm_vocab_size, args.transformer_attention_dim, padding_idx=151643)
        self.audio_embedding = nn.Embedding(self.vocab_size + 9, args.transformer_attention_dim, padding_idx=self.vocab_size + 3)
        # self.init_pre_nn(config)

        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx, do_cross_attention = False) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.dropout = nn.Dropout(p=self.encoder_drop_rate)
        self.out_fnn = nn.Linear(args.encoder_output_dim, self.vocab_size + 9)
        
        if self.encoder_criterion == 'ce':
            self.criterion = CrossEntropyLoss(ignore_index=self.vocab_size + 3)

    def forward(self, batch):
        # print("ar_ori")
        llm_hidden = batch['x']
        llm_hidden = self.embed_tokens(llm_hidden)
        llm_hidden_lens = batch['x_lens']
        y = batch['y']
        # print("before",y)
        y[y == IGNORE_ID] = self.vocab_size + 3
        y_lens = batch['y_lens']
        if 'prompt' in batch:
            prompt = batch['prompt']
            prompt[prompt == IGNORE_ID] = 0
            prompt = self.embed_tokens(prompt)
            prompt_lens = batch['prompt_lens']
        else: # default
            prompt = None# "Speaker: Brain, Language: English, Gender: Male."
        if 'prefix' in batch:
            prefix = batch['prefix']
            prefix[prefix == IGNORE_ID] = 0
            prefix = self.audio_embedding(prefix)
            prefix_lens = batch['prefix_lens']
            no_pref = False
        else:
            prefix = None
            no_pref = True
        # past_key_values = DynamicCache.from_legacy_cache(None)

        # text_ids: (batch_size, max_len)
        batch_size, max_len = y.size()

        # Create bos, sos and eos tokens
        bos_token = torch.full((batch_size, 1), self.vocab_size, dtype=torch.long, device=y.device)
        sos_token = torch.full((batch_size, 1), self.vocab_size + 1, dtype=torch.long, device=y.device)
        eos_token = torch.full((batch_size, 1), self.vocab_size + 2, dtype=torch.long, device=y.device)
        padding_token = torch.full((batch_size, 1), self.vocab_size + 3, dtype=torch.long, device=y.device)

        # prompt text embedding
        sopt_token = torch.full((1, 1), self.vocab_size + 4, dtype=torch.long, device=y.device).squeeze(0)
        eopt_token = torch.full((1, 1), self.vocab_size + 5, dtype=torch.long, device=y.device).squeeze(0)
        # prefix text embedding
        sop_token = torch.full((1,1), self.vocab_size + 6, dtype=torch.long, device=y.device).squeeze(0)
        eop_token = torch.full((1,1), self.vocab_size + 7, dtype=torch.long, device=y.device).squeeze(0)


        eot_token = torch.full((batch_size, 1), self.vocab_size + 8, dtype=torch.long, device=y.device)
        # Pass through pre_nn
        # llm_hidden = self.pre_nn_forward(llm_hidden, llm_hidden_lens)
        # [prefix, prompt, llm_hidden]
        # all_hidden = [h for h in [prefix, prompt, llm_hidden] if h is not None]
        # concat_llm_hidden = torch.cat(all_hidden, dim=-2)
        # concat_llm_hidden = self.in_fnn(concat_llm_hidden)
        # split_sizes = [hidden.shape[-2] for hidden in all_hidden]
        # llm_hiddens = torch.split(concat_llm_hidden, split_sizes, dim=-2)
        

        pad_embed = self.audio_embedding(padding_token[0])
        if not no_pref:
            pref_hidden = prefix
            max_pref_hidden_len = pref_hidden.shape[-2]
            sop_emb = self.audio_embedding(sop_token)
            eop_emb = self.audio_embedding(eop_token)
            # with open("/mnt/data/jsy/audio_proj/models/uni_omni_pretrain/f.txt","a") as f:
            #     f.write(str(batch['prefix'])+str(batch['prefix_lens'])+"\n")
            # print(batch['prefix'],batch['prefix_lens'])
            batch_pref_hidden = [torch.cat([sop_emb,pref_hidden[idx][:prefix_lens[idx]],eop_emb], dim=-2) for idx in range(pref_hidden.shape[0])]
            prefix_lens = prefix_lens+2
            max_pref_hidden_len = max_pref_hidden_len+2
            pref_hidden = torch.stack([torch.cat([batch_pref_hidden[batch_idx]]+(max_pref_hidden_len-prefix_lens[batch_idx])*[pad_embed], dim=-2) for batch_idx in range(llm_hidden.shape[0])], dim=0)

        if prompt is not None:
            prompt_hidden = prompt
            max_prompt_hidden_len = prompt_hidden.shape[-2]
            sopt_emb = self.audio_embedding(sopt_token)
            eopt_emb = self.audio_embedding(eopt_token)
            batch_prompt_hidden = [torch.cat([sopt_emb,prompt_hidden[idx][:prompt_lens[idx]],eopt_emb], dim=-2) for idx in range(prompt_hidden.shape[0])]
            prompt_lens = prompt_lens+2
            max_prompt_hidden_len = max_prompt_hidden_len+2
            prompt_hidden = torch.stack([torch.cat([batch_prompt_hidden[batch_idx]]+(max_prompt_hidden_len-prompt_lens[batch_idx])*[pad_embed], dim=-2) for batch_idx in range(llm_hidden.shape[0])], dim=0)
            
        # print(batch_size,llm_hidden.shape)
        # Concat bos embedding
        bos_emb = self.audio_embedding(bos_token)
        eot_emb= self.audio_embedding(eot_token)
        llm_hidden = torch.cat([bos_emb, llm_hidden, eot_emb], dim=1)
        llm_hidden_lens = llm_hidden_lens + 2

        # Concat prompt
        # llm_hidden = torch.cat([prompt_hidden, llm_hidden], dim=1)
        # llm_hidden_lens = llm_hidden_lens + prompt_lens

        # Create input x with sos token at the beginning
        x = torch.cat([sos_token, y], dim=1)  # (batch_size, max_len + 1)
        
        # Create output y with eos token at the end
        y = torch.cat([y, padding_token], dim=1)
        eos_positions = torch.arange(max_len + 1, device=y.device).expand(batch_size, max_len + 1) \
                        == y_lens.unsqueeze(1)
        y = y.masked_scatter(eos_positions, eos_token.expand_as(y)[eos_positions])

        # Embed the input sequence
        x_emb = self.audio_embedding(x)  # (batch_size, max_len + 1, d_model)

        # compute causal self attention masks
        input_lens = llm_hidden.size(1) + max_len + 1
        # input_mask = torch.zeros(batch_size, input_lens, input_lens, dtype=torch.bool, device=x_emb.device)
        # for i in range(batch_size):
        #     input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = True
        #     input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, :llm_hidden_lens[i]] = True
        #     input_mask[i, llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1, \
        #                 llm_hidden.size(1): llm_hidden.size(1) + y_lens[i] + 1] \
        #                 = subsequent_mask(y_lens[i] + 1, x_emb.device)

        # # Pass through the transformer
        # inputs_embeds = torch.cat([llm_hidden, x_emb], 1)
        # # llm_hidden = self.dropout(llm_hidden)
        # past_seen_tokens = 0
        # cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
        #                               device=inputs_embeds.device)
        # position_ids = cache_position.unsqueeze(0)
        # hidden_states = inputs_embeds
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min

        # encoder attention mask and position ids
        
        if no_pref:
            encoder_inputs_embeds_lens = prompt_lens
            encoder_inputs_embeds = prompt_hidden
            # print(encoder_inputs_embeds.shape)
        else:
            encoder_inputs_embeds_lens = [prompt_lens[batch_idx]+prefix_lens[batch_idx] for batch_idx in range(llm_hidden.shape[0])]
            max_encoder_inputs_embeds_lens = max(encoder_inputs_embeds_lens)
            encoder_inputs_embeds = torch.stack([torch.cat([batch_prompt_hidden[batch_idx],batch_pref_hidden[batch_idx]]+(max_encoder_inputs_embeds_lens-encoder_inputs_embeds_lens[batch_idx])*[pad_embed], dim=-2) for batch_idx in range(llm_hidden.shape[0])], dim=0)


        encoder_input_lens = llm_hidden.size(1) + max_len + 1
        encoder_input_mask = torch.zeros(batch_size, encoder_inputs_embeds.size(1) + input_lens, encoder_inputs_embeds.size(1) + input_lens, \
                                    dtype=torch.bool, device=x_emb.device)
        for i in range(batch_size):
            encoder_input_mask[i, :encoder_inputs_embeds_lens[i], :encoder_inputs_embeds_lens[i]] = True
            encoder_input_mask[i, :encoder_inputs_embeds_lens[i], encoder_inputs_embeds.size(1): encoder_inputs_embeds.size(1) + llm_hidden_lens[i]] = True
            encoder_input_mask[i, encoder_inputs_embeds.size(1): encoder_inputs_embeds.size(1) + llm_hidden_lens[i], :encoder_inputs_embeds_lens[i]] = True
            encoder_input_mask[i, encoder_inputs_embeds.size(1): encoder_inputs_embeds.size(1) + llm_hidden_lens[i], encoder_inputs_embeds.size(1): encoder_inputs_embeds.size(1) + llm_hidden_lens[i]] = True

            encoder_input_mask[i, encoder_inputs_embeds.size(1) + llm_hidden.size(1): encoder_inputs_embeds.size(1) + llm_hidden.size(1) + y_lens[i] + 1, :encoder_inputs_embeds_lens[i]] = True
            encoder_input_mask[i, encoder_inputs_embeds.size(1) + llm_hidden.size(1): encoder_inputs_embeds.size(1) + llm_hidden.size(1) + y_lens[i] + 1, \
                        encoder_inputs_embeds.size(1): encoder_inputs_embeds.size(1) + llm_hidden_lens[i]] = True
            encoder_input_mask[i, encoder_inputs_embeds.size(1) + llm_hidden.size(1): encoder_inputs_embeds.size(1) + llm_hidden.size(1) + y_lens[i] + 1, \
                        encoder_inputs_embeds.size(1) + llm_hidden.size(1): encoder_inputs_embeds.size(1) + \
                                                                llm_hidden.size(1) + y_lens[i] + 1] \
                        = subsequent_mask(y_lens[i] + 1, x_emb.device)
        # draw_mask(encoder_input_mask)
        hidden_states = torch.cat([encoder_inputs_embeds, llm_hidden, x_emb], 1)
        encoder_attention_mask =  ~(encoder_input_mask.unsqueeze(1)) * torch.finfo(hidden_states.dtype).min

        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + hidden_states.shape[1], \
                                    device=hidden_states.device)
        encoder_position_ids = cache_position.unsqueeze(0)
        # hidden_states = encoder_inputs_embeds
        encoder_position_embeddings = self.rotary_emb(hidden_states, encoder_position_ids)
        
            

        # print(hidden_states.shape,attention_mask.shape)
        # print(attention_mask)
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                past_key_value= None,# past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=encoder_position_embeddings,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        encoder_out = hidden_states[:, encoder_inputs_embeds.size(1) + llm_hidden.size(1):]

        # Project to vocabulary size
        logits = self.out_fnn(encoder_out)
        # print(logits)
        # print(hidden_states.shape,encoder_out.shape,logits.shape,y.shape)
        # print("after",y)
        if self.encoder_criterion == 'ce':
            loss = self.criterion(logits, y)

        # if self.training:
        #     self.reporter.log_loss('loss', float(loss))

        return loss
    
    def transformer_infer(self, encoder_inputs_embeds, encoder_cache_position):
        position_ids = encoder_cache_position.unsqueeze(0)
        hidden_states = encoder_inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        next_decoder_cache = None
        # encoder_position_ids = encoder_cache_position.unsqueeze(0)
        # encoder_position_embeddings = self.rotary_emb(encoder_inputs_embeds, encoder_position_ids)
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                # encoder_hidden_states=encoder_inputs_embeds,
                # encoder_attention_mask=None,
                # encoder_position_ids=encoder_position_ids,
                # encoder_position_embeddings=encoder_position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]
        return hidden_states
            
    def infer(self, hidden, prompt, prefix, top_k, penalty_window_size, penalty, max_tokens=1000):
        hidden = self.embed_tokens(hidden)
        no_pref = True if prefix is None else False

        # Create bos, sos and eos tokens
        bos_token = torch.full((1, 1), self.vocab_size, dtype=torch.long, device=hidden.device)
        sos_token = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        eos_token = torch.full((1, 1), self.vocab_size + 2, dtype=torch.long, device=hidden.device)
        padding_token = torch.full((1, 1), self.vocab_size + 3, dtype=torch.long, device=hidden.device)

        # prompt text embedding
        sopt_token = torch.full((1, 1), self.vocab_size + 4, dtype=torch.long, device=hidden.device)
        eopt_token = torch.full((1, 1), self.vocab_size + 5, dtype=torch.long, device=hidden.device)
        # prefix text embedding
        sop_token = torch.full((1,1), self.vocab_size + 6, dtype=torch.long, device=hidden.device)
        eop_token = torch.full((1,1), self.vocab_size + 7, dtype=torch.long, device=hidden.device)

        eot_token = torch.full((1, 1), self.vocab_size + 8, dtype=torch.long, device=hidden.device)
        
        # Pass through pre_nn
        # all_hidden = [h for h in [prefix, prompt, hidden] if h is not None]
        # concat_llm_hidden = torch.cat(all_hidden, dim=-2)
        # concat_llm_hidden = self.in_fnn(concat_llm_hidden)
        # split_sizes = [hidden.shape[-2] for hidden in all_hidden]
        # llm_hiddens = torch.split(concat_llm_hidden, split_sizes, dim=-2)
        # print([i.shape for i in llm_hiddens])
        # hidden = llm_hiddens[-1]

        # Concat bos embedding
        bos_emb = self.audio_embedding(bos_token)
        # hidden = torch.cat([bos_emb, hidden], dim=1)
        eot_emb= self.audio_embedding(eot_token)
        hidden = torch.cat([bos_emb, hidden, eot_emb], dim=1)
        en_hidden = hidden.clone()

        pad_embed = self.audio_embedding(padding_token[0])
        if not no_pref:
            # pref_hidden = llm_hiddens[0]
            pref_hidden = self.audio_embedding(prefix)
            sop_emb = self.audio_embedding(sop_token)
            eop_emb = self.audio_embedding(eop_token)
            pref_hidden = torch.cat([sop_emb, pref_hidden, eop_emb], dim=1)
            en_hidden = torch.cat([pref_hidden, en_hidden], dim=1)

        if prompt is not None:
            prompt_hidden = self.embed_tokens(prompt)
            sopt_emb = self.audio_embedding(sopt_token)
            eopt_emb = self.audio_embedding(eopt_token)
            prompt_hidden = torch.cat([sopt_emb, prompt_hidden, eopt_emb], dim=1)
            en_hidden = torch.cat([prompt_hidden, en_hidden], dim=1)
        
        inputs_embeds = hidden
        encoder_inputs_embeds = en_hidden
        print(hidden.shape, en_hidden.shape)
        past_seen_tokens = 0
        # cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
        #                               device=inputs_embeds.device)
        # hidden_states = self.transformer_infer(inputs_embeds, cache_position)

        # init generated tokens
        cur_token = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        # generate tokens
        all_tokens = generated_tokens[0]
        for i in range(max_tokens):
            # inputs_embeds = torch.cat([inputs_embeds,self.audio_embedding(cur_token)], dim=1)
            encoder_inputs_embeds = torch.cat([encoder_inputs_embeds,self.audio_embedding(cur_token)], dim=1)
            # cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                        #   device=inputs_embeds.device)
            encoder_cache_position = torch.arange(past_seen_tokens, past_seen_tokens + encoder_inputs_embeds.shape[1], \
                                        device=encoder_inputs_embeds.device)
            
            hidden_states = self.transformer_infer(encoder_inputs_embeds, encoder_cache_position)
            hidden_states = self.norm(hidden_states)
            

            # Project to vocabulary size
            logits = self.out_fnn(hidden_states)
            
            # apply penalty
            if penalty_window_size > 0:
                for token in set(generated_tokens[0][-penalty_window_size:]):
                    logits[:, :, token] /= penalty

            # top k sampling
            output = logits.squeeze(0).squeeze(0)
            output = output[-1]
            probs = torch.nn.functional.softmax(output, dim=-1)
            
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
            next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_token = next_token_id

            # eos
            if next_token_id == self.vocab_size + 2:
                break
            all_tokens = torch.cat([all_tokens, next_token_id[0]], dim=-1)
        # print(all_tokens)
        return all_tokens[1:]
