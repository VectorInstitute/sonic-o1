import os
import torch
import torch.nn as nn
import re
from transformers import WhisperConfig, WhisperModel, WhisperProcessor, WhisperPreTrainedModel


class WhisperAudioTower(nn.Module):
    def __init__(self, audio_tower_config, delay_load=False):
        super().__init__()
        self.is_loaded = False

        self.audio_split_type_dim = 4
        self.local_files_only=True

        self.config = audio_tower_config.audio_config
        self.audio_tower = WhisperModel._from_config(self.config, attn_implementation="eager").encoder

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs[0]

        return audio_features

    def forward(self, audios):
        if audios.dim() == self.audio_split_type_dim:
            audio_features = []
            for k in range(audios.shape[0]):
                audio = audios[k,:,:,:]
                audio_forward_out = self.audio_tower(audio.to(device=self.device, dtype=self.dtype), output_hidden_states=True,return_dict=True)
                audio_feature = self.feature_select(audio_forward_out).to(audio.dtype)
                audio_features.append(audio_feature)
            audio_features = torch.cat(audio_features, dim=1)
        else:
            audio_forward_outs = self.audio_tower(audios.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audios.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.audio_size // self.config.patch_size) ** 2

def build_whisper_tower(audio_tower_cfg, **kwargs):
    return WhisperAudioTower(audio_tower_cfg, **kwargs)


class WhisperAudioAligner(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # print(config)
        self.decoder = WhisperAudioDecoder(config)
        projector_type = getattr(config, 'whisper_projector_type', 'linear')
        if projector_type == 'linear':
            projector = nn.Linear(config.whisper_hidden_size, config.hidden_size)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.whisper_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                projector = nn.Sequential(*modules)
        self.projector = projector
        self.query_tokens = nn.Parameter(torch.zeros(1, config.whisper_query_tokens_size, config.whisper_hidden_size))
    def forward(self, encoder_output ,**kwargs):
        query_tokens = self.query_tokens.expand(encoder_output.shape[0], -1, -1)
        decoder_output = self.decoder(encoder_output = encoder_output,query_tokens= query_tokens,**kwargs)
        projector_output = self.projector(decoder_output)
        return projector_output

class WhisperAudioDecoder(nn.Module):
    def __init__(self, audio_tower_config, delay_load=False):
        super().__init__()
        self.audio_decoder = WhisperModel._from_config(audio_tower_config.audio_config, attn_implementation="eager").decoder
    
    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.last_hidden_state
        return audio_features

    # @torch.no_grad()
    def forward(self, encoder_output, query_tokens):
        audio_decoder_outs = self.audio_decoder(inputs_embeds = query_tokens.to(device=self.device, dtype=self.dtype),
                                                encoder_hidden_states=encoder_output,
                                                output_hidden_states=True,
                                                return_dict=True,
                                                )
        audio_features = self.feature_select(audio_decoder_outs).to(encoder_output.dtype)
        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_decoder.dtype

    @property
    def device(self):
        return self.audio_decoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_decoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.audio_size // self.config.patch_size) ** 2

def build_whisper_aligner(audio_tower_cfg, **kwargs):
    return WhisperAudioAligner(audio_tower_cfg, **kwargs)
