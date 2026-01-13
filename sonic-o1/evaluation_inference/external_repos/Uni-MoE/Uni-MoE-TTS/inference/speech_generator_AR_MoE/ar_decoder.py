import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Union
from transformers import Qwen2Config,AutoConfig
from .modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
from transformers.cache_utils import DynamicCache

from .masks import *

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
        self.idim = args.idim
        self.odim = args.odim
        self.encoder_pre_norm_type = args.encoder_pre_norm_type
        self.encoder_drop_rate = args.encoder_drop_rate
        self.encoder_criterion = args.encoder_criterion
        self.encoder_upsample_rate = args.encoder_upsample_rate
        self.reporter = None
        self.prefix_len = 10
        self.config = args

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
                            attention_dropout=args.transformer_dropout_rate,
                            # moe settings
                            num_experts = 4,
                            ep_size = 1,
                            capacity_factor = 1.5,
                            moe_dp = True,
                            )
        qwenconfig = AutoConfig.from_pretrained(args.qwenconfig)#args.qwenconfig
        self.qwenvl_embed_tokens = nn.Embedding(qwenconfig.vocab_size, qwenconfig.hidden_size, qwenconfig.pad_token_id)
        self.in_fnn = nn.Linear(self.idim, args.transformer_attention_dim)
        self.audio_embedding = nn.Embedding(self.vocab_size + 9, args.transformer_attention_dim, padding_idx=self.vocab_size + 3)

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
        llm_hidden = batch['x']
        llm_hidden = self.qwenvl_embed_tokens(llm_hidden)
        llm_hidden = self.in_fnn(llm_hidden)
        llm_hidden_lens = batch['x_lens']
        y = batch['y']
        y[y == IGNORE_ID] = self.vocab_size + 3
        y_lens = batch['y_lens']
        if 'prompt' in batch:
            prompt = batch['prompt']
            prompt[prompt == IGNORE_ID] = 0
            prompt = self.qwenvl_embed_tokens(prompt)
            prompt = self.in_fnn(prompt)
            prompt_lens = batch['prompt_lens']
        else: # default
            prompt = None
        if 'prefix' in batch:
            prefix = batch['prefix']
            prefix[prefix == IGNORE_ID] = 0
            prefix = self.audio_embedding(prefix)
            prefix_lens = batch['prefix_lens']
            no_pref = False
        else:
            prefix = None
            no_pref = True

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
        
        pad_embed = self.audio_embedding(padding_token[0])
        if not no_pref:
            pref_hidden = prefix
            max_pref_hidden_len = pref_hidden.shape[-2]
            sop_emb = self.audio_embedding(sop_token)
            eop_emb = self.audio_embedding(eop_token)
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
            
        # Concat bos embedding
        bos_emb = self.audio_embedding(bos_token)
        eot_emb= self.audio_embedding(eot_token)
        llm_hidden = torch.cat([bos_emb, llm_hidden, eot_emb], dim=1)
        llm_hidden_lens = llm_hidden_lens + 2

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

        # encoder attention mask and position ids
        if no_pref:
            encoder_inputs_embeds_lens = prompt_lens
            encoder_inputs_embeds = prompt_hidden
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
        encoder_position_embeddings = self.rotary_emb(hidden_states, encoder_position_ids)
        
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

        if self.encoder_criterion == 'ce':
            loss = self.criterion(logits, y)

        return loss
    
    def transformer_infer(self, encoder_inputs_embeds, encoder_cache_position):
        position_ids = encoder_cache_position.unsqueeze(0)
        hidden_states = encoder_inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        next_decoder_cache = None
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
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
        hidden = self.qwenvl_embed_tokens(hidden)
        hidden = self.in_fnn(hidden)
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

        # Concat bos embedding
        bos_emb = self.audio_embedding(bos_token)
        eot_emb= self.audio_embedding(eot_token)
        hidden = torch.cat([bos_emb, hidden, eot_emb], dim=1)
        en_hidden = hidden.clone()

        pad_embed = self.audio_embedding(padding_token[0])
        if not no_pref:
            pref_hidden = self.audio_embedding(prefix)
            sop_emb = self.audio_embedding(sop_token)
            eop_emb = self.audio_embedding(eop_token)
            pref_hidden = torch.cat([sop_emb, pref_hidden, eop_emb], dim=1)
            en_hidden = torch.cat([pref_hidden, en_hidden], dim=1)

        if prompt is not None:
            prompt = self.qwenvl_embed_tokens(prompt)
            prompt_hidden = self.in_fnn(prompt)
            sopt_emb = self.audio_embedding(sopt_token)
            eopt_emb = self.audio_embedding(eopt_token)
            prompt_hidden = torch.cat([sopt_emb, prompt_hidden, eopt_emb], dim=1)
            en_hidden = torch.cat([prompt_hidden, en_hidden], dim=1)
        
        inputs_embeds = hidden
        encoder_inputs_embeds = en_hidden
        past_seen_tokens = 0

        # init generated tokens
        cur_token = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.vocab_size + 1, dtype=torch.long, device=hidden.device)
        # generate tokens
        all_tokens = generated_tokens[0]
        for i in range(max_tokens):
            encoder_inputs_embeds = torch.cat([encoder_inputs_embeds,self.audio_embedding(cur_token)], dim=1)
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

        return all_tokens[1:]
    
    def extract_tts_pos(self,input_ids):
        sosp_pos = []
        eop_pos = []
        eosp_pos = []

        for batch_input_ids in input_ids:
            sosp_pos.append(torch.where(batch_input_ids==self.config.speech_start_token_id)[0])
            eop_pos.append(torch.where(batch_input_ids==self.config.speech_prompt_token_id)[0])
            eosp_pos.append(torch.where(batch_input_ids==self.config.speech_end_token_id)[0])
        return sosp_pos, eop_pos, eosp_pos

    def extract_speech_ids(self,sosp_pos, eop_pos, eosp_pos, speech_splits, input_ids, codes, split_tokens):
        all_prompt = []
        all_prefix = []
        all_text = []
        code_pos = 0
        for b_sosp_pos, b_eop_pos, b_eosp_pos, b_speech_splits, b_input_ids in zip(sosp_pos, eop_pos, eosp_pos, speech_splits, input_ids):
            assert len(b_sosp_pos) == len(b_eop_pos) == len(b_eosp_pos)
            speech_idx = 0
            split_idx = 0
            for speech_idx in range(len(b_sosp_pos)):
                prompt = b_input_ids[b_sosp_pos[speech_idx]+1:b_eop_pos[speech_idx]]
                split_chunk = []
                split_pos = []
                while len(b_speech_splits)>0 and split_idx<len(b_speech_splits) and b_speech_splits[split_idx] > b_eop_pos[speech_idx] and b_speech_splits[split_idx] < b_eosp_pos[speech_idx]:
                    split_pos.append(b_speech_splits[split_idx])
                    split_idx+=1
                    if split_idx>=len(b_speech_splits):
                        break
                # first cut
                all_prefix.append(None)
                all_prompt.append(prompt)
                code_pos+=1
                if len(split_pos):
                    all_text.append(b_input_ids[b_eop_pos[speech_idx]+1:split_pos[0]])
                else:
                    all_text.append(b_input_ids[b_eop_pos[speech_idx]+1:b_eosp_pos[speech_idx]])
                # other cuts
                for cut_idx,_ in enumerate(split_pos):
                    all_prompt.append(prompt)
                    if codes != None:
                        last_code_pos = 0
                        for scid,sc in enumerate(codes[code_pos-1]):
                            if sc!= IGNORE_INDEX:
                                last_code_pos = scid
                        last_code_pos+=1
                        if last_code_pos-self.prefix_len>=0:
                            all_prefix.append(codes[code_pos-1][last_code_pos-self.prefix_len:last_code_pos])
                        else:
                            all_prefix.append(torch.cat([codes[code_pos-1][:last_code_pos],torch.LongTensor([IGNORE_INDEX]*(self.prefix_len-last_code_pos)).to(device=codes.device)],dim=0))
                    else:
                        all_prefix.append(None)
                    if cut_idx==len(split_pos)-1:
                        all_text.append(b_input_ids[split_pos[cut_idx]:b_eosp_pos[speech_idx]])
                    else:
                        all_text.append(b_input_ids[split_pos[cut_idx]:split_pos[cut_idx+1]])
                    code_pos+=1
        batch = {}
        tmpd = {}
        batch["prompt_ids"] = all_prompt
        prompt = torch.nn.utils.rnn.pad_sequence(all_prompt,
                                                batch_first=True,
                                                padding_value=0)
        prompt = prompt.to(device=input_ids.device)
        prompt_len = [int(x.shape[0]) if x != None else 0 for x in all_prompt]
        tmpd["prompt_lens"] = torch.LongTensor(prompt_len).to(device=prompt.device)
        batch["prompt"] = self.qwenvl_embed_tokens(prompt).to(device=prompt.device)
        batch["prompt_lens"] = torch.LongTensor(prompt_len).to(device=prompt.device)

        if codes != None and sum([pr!=None for pr in all_prefix]):
            batch["prefix"] = torch.stack([pref if pref != None else torch.LongTensor([IGNORE_INDEX]*self.prefix_len).to(device=codes.device) for pref in all_prefix],dim=0)
            batch["prefix_lens"] = torch.LongTensor([self.prefix_len if pref != None else 0 for pref in all_prefix]).to(device=codes.device)
            tmpd["prefix"] = batch["prefix"]
            tmpd["prefix_lens"] = batch["prefix_lens"]
        # pad text
        all_text = [torch.cat([sep_text[:-2],torch.LongTensor([13]).to(device=prompt.device)]) if sep_text[-1] == torch.LongTensor([291]).to(device=prompt.device) and sep_text[-1] == torch.LongTensor([5102]).to(device=prompt.device) else sep_text for sep_text in all_text]
        all_text = [sep_text if sep_text[-1] in split_tokens else torch.cat([sep_text,torch.LongTensor([13]).to(device=prompt.device)])  for sep_text in all_text]
        text = torch.nn.utils.rnn.pad_sequence(all_text,
                                                batch_first=True,
                                                padding_value=0)
        text_len = [int(x.shape[0]) if x != None else 0 for x in all_text]
        tmpd["x_lens"] = torch.LongTensor(text_len).to(device=text.device)
        batch["x"] = self.qwenvl_embed_tokens(text).to(device=text.device)
        batch["x_ids"] = all_text
        batch["x_lens"] = torch.LongTensor(text_len).to(device=text.device)

        return batch,tmpd

    def select_split2(self, out_seq, split_tokens, number_tokens,maxcutlen):
        # use . to split
        minlen = 5
        maxlen = maxcutlen
        split_idx = []
        for b_os in out_seq:
            split_b = []
            now_prompt = -1
            now_start = -1
            for idx,i in enumerate(b_os):
                if i == self.config.speech_prompt_token_id: #151667:
                    now_prompt = idx
                    now_start = idx
                    now_end = now_start
                if i == self.config.speech_end_token_id:#151666:
                    now_prompt = -1
                    now_start = -1
                if now_prompt>0 and now_start>0:
                    if idx - now_start > maxlen and b_os[idx+1] not in number_tokens and b_os[idx+1] not in split_tokens: 
                        now_end=idx+1
                        split_b.append(now_end)
                        now_start = now_end
                    if i in split_tokens and b_os[idx+1] not in number_tokens and b_os[idx+1] != self.config.speech_end_token_id:
                        now_end=idx+1
                        if now_end - now_start > minlen:
                            split_b.append(now_end)
                            now_start = now_end
            split_bt = torch.LongTensor(split_b)
            split_bt = split_bt.to(device=out_seq.device)
            split_idx.append(split_bt)
        return split_idx

    @torch.no_grad()
    def generate_from_tokens(
        self,
        out_seq: Optional[torch.Tensor] = None,
        split_tokens = None,
        number_tokens = None,
        maxtoklen = None,
        maxcutlen = None,
        **kwargs,
    ):
        o_device = out_seq.device
        if out_seq.shape[-1]>=maxtoklen and int(self.config.speech_end_token_id) not in out_seq[0].tolist():
            print("too long cut...")
            out_seq = out_seq[0].tolist()
            bad = out_seq[-1]
            i = len(out_seq)-1
            while out_seq[i]==bad:
                i-=1
            out_seq = out_seq[:i+2]+[13, int(self.config.speech_end_token_id), 151645]
            out_seq = torch.LongTensor([out_seq]).to(device=o_device)

        sosp_pos, eop_pos, eosp_pos = self.extract_tts_pos(out_seq)
        split_idx = self.select_split2(out_seq,split_tokens,number_tokens,maxcutlen)
        batch,tmpd = self.extract_speech_ids(sosp_pos, eop_pos, eosp_pos, split_idx, out_seq, None, split_tokens)
        tmpd['prompt'] = batch["prompt"]
        tmpd['x'] = batch["x"]
        tmpd['prompt_ids'] = batch["prompt_ids"]
        tmpd['x_ids'] = batch["x_ids"]
        ar_preds = []
        tmp_pred = None
        for i in range(tmpd["x"].shape[0]):
            tmp_pred = self.infer(tmpd["x_ids"][i].unsqueeze(0), tmpd["prompt_ids"][i].unsqueeze(0), None, top_k=1, penalty_window_size=-1, penalty=1.0)
            ar_preds.append(tmp_pred)
        
        return ar_preds,split_idx

def load_moetts(ckpt_path):
    class ModelArguments:
        # ar config parts
        idim = 2048 # 2048 输入的维度 qwen llm的特征维度
        odim = 4096 # 1024 词表维度 tokenizer的词表大小
        encoder_pre_norm_type = "ln" # "ln" 似乎没用到
        encoder_drop_rate = 0.1  # 0.1 drop out的程度 因为没有NAR所以用不到
        encoder_criterion = "ce" # "ce" 损失函数类型，不改动
        encoder_upsample_rate = 9 # 9 没有做upsample所以也用不到
        transformer_attention_dim = 896 # 896 llama hidden_size 4096 
        transformer_linear_units = 4864 # 4864 llama intermediate_size 11008
        transformer_num_blocks = 24 # 4 llama num_hidden_layers 32 
        transformer_attention_heads = 14 # 14 llama num_attention_heads 32
        transformer_kv_heads = 2 
        transformer_dropout_rate = 0.1  # 0.1 
        encoder_output_dim = 896 # 896 输出词表大小
        do_cross_attention = True
        cross_attention_layer_num = 6
        audio_mode= "tts_only_sft"
        speech_generator_type= "ar_ori_v2"
        llm_vocab_size = 151936
        speech_start_token_id = 151668
        speech_prompt_token_id = 151670
        speech_end_token_id = 151669
        speech_split_token_id = 151671
        qwenconfig = "./qwen_config"
    config = ModelArguments()
    model = LLM2TTSCodecAR(config)
    model_sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model_sd = {k.replace("speech_generator.",""): v for k,v in model_sd.items()}
    model.load_state_dict(model_sd, strict=True)
    return model