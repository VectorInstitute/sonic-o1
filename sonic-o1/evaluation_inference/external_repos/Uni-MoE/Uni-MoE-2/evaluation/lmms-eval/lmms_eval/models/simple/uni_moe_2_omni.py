import copy
import os
import sys
import warnings
from typing import List, Optional, Tuple, Union
import json
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from PIL import Image


# Add the root path to sys.path for importing uni_moe modules
from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
from uni_moe.model import deepspeed_moe_inference_utils
from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
from uni_moe.qwen_vl_utils import process_mm_info
from moviepy import VideoFileClip
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

SYSTEM_PROMPT = "You are Uni-MoE-2, a helpful multi-modal model. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines."

@register_model("uni_moe_2_omni")
class UniMoE_2_Omni(lmms):
    """
    Uni-MoE_2_Omni Model for Multi-modal Evaluation
    """

    def __init__(
        self,
        pretrained: str = "HIT-TMG/Uni-MoE-2.0-Omni",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        think_mode: Optional[bool] = False,
        batch_size: Optional[Union[int, str]] = 1,
        use_audio_in_video: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        self._processor = Qwen2VLProcessor.from_pretrained(pretrained)
        self._model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(pretrained, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        
        # Configure processor
        self._processor.data_args = self._model.config

        self._config = self._model.config
        self._tokenizer = self._processor.tokenizer
        
        self.model.eval()
        self.batch_size_per_gpu = int(batch_size)
        self.use_audio_in_video = use_audio_in_video
        self.think_mode = think_mode
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """Tokenize a string"""
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def meta_form(self, sources, image_folder="", audio_folder="", video_folder=""):
        """Convert data format to messages format"""
        messages = []
        images = audios = videos = None
        images_idx = audios_idx = videos_idx = 0
        
        if sources.get("images"):  ## Important Remember to change.....
            images = sources["images"]
            if type(images) == str: images = [images]
            if isinstance(images[0], str):
                images = [os.path.join(image_folder, image) for image in images]
            images_len = len(images)
        if sources.get("audios"): 
            audios = sources["audios"]
            if type(audios) == str: audios = [audios]
            audios = [os.path.join(audio_folder, audio) for audio in audios]
            audios_len = len(audios)
        if sources.get("videos"): 
            videos = sources["videos"]
            if type(videos) == str: videos = [videos]
            assert isinstance(videos[0], str)
            if videos[0][0] != "[":
                videos = [os.path.join(video_folder, video) for video in videos]
            else:
                new_videos = []
                for video in videos:
                    frames_list = json.loads(video)
                    frames_list = [os.path.join(self.data_args.video_folder, frame) for frame in frames_list]
                    new_videos.append(frames_list)
                videos = new_videos
            videos_len = len(videos)
            
        convs = sources["conversations"]        
        for conv in convs:
            text = conv["value"]
            tag_count = text
            if conv["from"] == "human":
                conv_dic = {'role': 'user', "content":[{"type": "text", "text": text}]}
            elif conv["from"] == "system":
                conv_dic = {'role': 'system', "content":[{"type": "text", "text": text}]}
            else:
                conv_dic = {'role': 'assistant', "content":[{"type": "text", "text": text}]}
                
            while images is not None and "<image>" in tag_count:
                assert images_len > images_idx
                conv_dic["content"].append({"type": "image", "image": images[images_idx]})
                tag_count = tag_count.replace("<image>", "", 1)
                images_idx += 1
                
            while audios is not None and "<audio>" in tag_count:
                assert audios_len > audios_idx
                conv_dic["content"].append({"type": "audio", "audio": audios[audios_idx]})
                tag_count = tag_count.replace("<audio>", "", 1)
                audios_idx += 1
                
            while videos is not None and "<video>" in tag_count:
                assert videos_len > videos_idx
                conv_dic["content"].append({"type": "video", "video": videos[videos_idx]})
                tag_count = tag_count.replace("<video>", "", 1)
                videos_idx += 1
                
            messages.append(conv_dic)
            
        messages = [messages]  # batch size = 1
        return messages

    def initial_input(self, line, image_folder="", audio_folder="", video_folder="", do_print=False):
        """Process input data into model format"""
        line_copy = copy.deepcopy(line)
        query = line_copy["conversations"][0]["value"] 
        modality_prefixes = []
        
        if line_copy.get("images", None) and "<image>" not in query:
            modality_prefixes.append("<image>")
        if line_copy.get("audios", None) and "<audio>" not in query:
            modality_prefixes.append("<audio>")
        if line_copy.get("videos", None) and "<video>" not in query:
            modality_prefixes.append("<video>")
            
        if modality_prefixes:
            query = "\n".join(modality_prefixes) + "\n" + query

        line_copy["conversations"][0]["value"] = query
        if self.think_mode:
            line_copy["conversations"] = [{"from": "system", "value": SYSTEM_PROMPT}] + line_copy["conversations"]
        
        messages = self.meta_form(line_copy)
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, no=True
        )
        
        for pi, pr in enumerate(prompt):
            prompt[pi] = prompt[pi].replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>").replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>").replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            
        image_inputs, video_inputs, audio_inputs = process_mm_info(messages)
        
        inputs = self._processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "second_grid_ts" in inputs:
            inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
            del inputs["second_grid_ts"]
        
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        return inputs

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            
            gen_kwargs = all_gen_kwargs[0]
            
            # Set default values for until and max_new_tokens
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
            until = [item for item in until if item != "\n\n"]
            
            if isinstance(contexts, tuple):
                contexts = list(contexts)
                
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
            
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            
            # try:
            for i, context in enumerate(contexts):
                target_messages = dict()
                target_messages["images"] = []
                target_messages["videos"] = []
                target_messages["audios"] = []
                message = [{"from": "human", "value": context}]
                target_messages["conversations"] = message
                
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            target_messages["videos"].append(visual)
                            # 加载音频
                            if self.use_audio_in_video and self._check_if_video_has_audio(visual):
                                target_messages["audios"].append(visual)
                            
                        elif isinstance(visual, Image.Image): # Handle both single and multiple images
                            target_messages["images"].append(visual)
                        elif isinstance(visual, list): # Video Frame list
                            target_messages["videos"].append(json.dumps(visual))
                        elif isinstance(visual, dict): # Video
                            import tempfile
                            import numpy as np
                            import torchaudio
                            audio_array = visual.get('array', None)
                            sampling_rate = visual.get('sampling_rate', None)
                            if isinstance(audio_array, np.ndarray):
                                audio_np = audio_array
                            else:
                                audio_np = np.array(audio_array)
                    
                            # Convert to torch tensor with shape (channels, samples)
                            if audio_np.ndim == 1:
                                audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, N)
                            elif audio_np.ndim == 2:
                                # Accept (N, C) or (C, N)
                                if audio_np.shape[0] < audio_np.shape[1]:
                                    audio_np = audio_np.T
                                audio_tensor = torch.from_numpy(audio_np).float()
                            else:
                                raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

                            # Resample to 16kHz if needed
                            target_sr = 16000
                            if sampling_rate != target_sr:
                                audio_tensor = torchaudio.functional.resample(audio_tensor, sampling_rate, target_sr)
                            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            torchaudio.save(temp_file.name, audio_tensor, target_sr, encoding="PCM_S", bits_per_sample=16)
        
                            target_messages["audios"].append(temp_file.name)

            inputs = self.initial_input(target_messages).to(device=self._model.device)
            for k, v in inputs.items():
                if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
                    inputs[k] = v.to(dtype=torch.bfloat16)
            # Set generation parameters
            if self.think_mode: # fixed generation parameters for think mode
                gen_kwargs["max_new_tokens"] = 16384
                gen_kwargs["temperature"] = 1.0
                gen_kwargs["do_sample"] = True
                gen_kwargs["num_beams"] = 1
            else:
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 32
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1
            
            # print(gen_kwargs)
            del gen_kwargs["until"]
            with torch.no_grad():
                output_ids = self.model.generate(
                **inputs,
                use_cache=True,
                pad_token_id=self._processor.tokenizer.eos_token_id,
                **gen_kwargs
                )
                
            text_outputs = self._processor.batch_decode(output_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip()
                
            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), text_outputs)
            pbar.update(1)
            
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for Uni-MoE-2")


    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Uni-MoE-2")