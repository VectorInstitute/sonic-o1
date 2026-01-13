import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers import AutoConfig
import torch.distributed as dist
import loralib
from loralib import LoRALayer
import math

class Qwen2MLPLoRA(nn.Module, LoRALayer):
    def __init__(self, config):
        super().__init__()
        LoRALayer.__init__(self, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, merge_weights=False)
        #self.pretraining_tp = config.pretraining_tp

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.init_lora()
    
    def forward_no_lora(self, x):
        # print("no lora")
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x):
        gate_project = self.gate_proj(x)
        if self.r > 0 and not self.merged:
            gate_project += self.gate_proj_lora_B(self.gate_proj_lora_A(self.lora_dropout(x))) * self.scaling
        gate_project = self.act_fn(gate_project)
        up_project = self.up_proj(x)
        if self.r > 0 and not self.merged:
            up_project += self.up_proj_lora_B(self.up_proj_lora_A(self.lora_dropout(x))) * self.scaling
        intermediate_states = gate_project * up_project
        down_project = self.down_proj(intermediate_states)
        if self.r > 0 and not self.merged:
            down_project += self.down_proj_lora_B(self.down_proj_lora_A(self.lora_dropout(intermediate_states))) * self.scaling
        return down_project
        #print("no lora")
        #return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def init_lora(self):
        if self.r > 0:
            self.gate_proj_lora_A = nn.Linear(self.hidden_size, self.r, bias=False)
            self.gate_proj_lora_B = nn.Linear(self.r, self.intermediate_size, bias=False)
            self.up_proj_lora_A = nn.Linear(self.hidden_size, self.r, bias=False)
            self.up_proj_lora_B = nn.Linear(self.r, self.intermediate_size, bias=False)
            self.down_proj_lora_A = nn.Linear(self.intermediate_size, self.r, bias=False)
            self.down_proj_lora_B = nn.Linear( self.r, self.hidden_size,bias=False)
            self.scaling = self.lora_alpha / self.r

        if hasattr(self, 'gate_proj_lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.gate_proj_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_proj_lora_B.weight)

        if hasattr(self, 'up_proj_lora_A'):
            nn.init.kaiming_uniform_(self.up_proj_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_lora_B.weight)

        if hasattr(self, 'down_proj_lora_A'):
            nn.init.kaiming_uniform_(self.down_proj_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj_lora_B.weight)
    
    def merge_lora(self):
        self.gate_proj.weight.data += (self.gate_proj_lora_B.weight @ self.gate_proj_lora_A.weight) * self.scaling
        self.down_proj.weight.data += (self.down_proj_lora_B.weight @ self.down_proj_lora_A.weight) * self.scaling
        self.up_proj.weight.data += (self.up_proj_lora_B.weight @ self.up_proj_lora_A.weight) * self.scaling

        nn.init.kaiming_uniform_(self.gate_proj_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj_lora_B.weight)
        nn.init.kaiming_uniform_(self.up_proj_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj_lora_B.weight)
        nn.init.kaiming_uniform_(self.down_proj_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj_lora_B.weight)

