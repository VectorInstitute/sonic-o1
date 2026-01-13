# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Qwen2-VL.
"""

from typing import List, Union
import os
import torch
import math
import numpy as np
from transformers.feature_extraction_utils import BatchFeature
try:
    from transformers.image_utils import ImageInput, VideoInput
except:
    from transformers.image_utils import ImageInput
    from transformers.video_utils import VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from uni_moe.vision_utils import process_unires_image, process_video_with_decord, process_unires_video
from PIL import Image
import time
# import librosa
# import soundfile

# Import and register SigLipImageProcessor
try:
    from uni_moe.model.visual_encoder.siglip_encoder import SigLipImageProcessor
    from transformers import AutoProcessor
    AutoProcessor.register(SigLipImageProcessor, "siglip_image_processor")
except ImportError:
    pass

IGNORE_INDEX = -100

logger = logging.get_logger(__name__)


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2VLProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2VLProcessor.__call__`] and [`~Qwen2VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    audio_processor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self,image_processor=None, audio_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, audio_processor, tokenizer, chat_template=chat_template)
        

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        audios = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # if images is not None:
        #     image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
        #     image_grid_thw = image_inputs["image_grid_thw"]
        #     print(image_inputs["pixel_values"])
        # else:
        #     image_inputs = {}
        #     image_grid_thw = None

        # if videos is not None:
        #     videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
        #     video_grid_thw = videos_inputs["video_grid_thw"]
        # else:
        #     videos_inputs = {}
        #     video_grid_thw = None
        image_inputs = {}
        if images is not None: # 输入 PIL
            pixel_values = []
            image_grid_thw = []
            for image in images:
                ###
                # image = image.resize((512, 512), Image.Resampling.LANCZOS)
                c_pixel_values, c_image_grid_thw = process_unires_image(image, self.image_processor, self.data_args.image_grid_pinpoints)
                pixel_values.append(c_pixel_values)
                image_grid_thw.append(c_image_grid_thw) # 1, 3
            image_inputs["pixel_values"] = torch.cat(pixel_values, dim=0) # seq_len, 3, 384, 384
            image_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0) # x, 3 [1, h, w]
        else:
            image_grid_thw = None

        videos_inputs = {}
        if videos is not None: # 输入 string
            pixel_values_video = []
            video_grid_thw = []
            video_sample_fps_list = []
            
            for video in videos:    
                if isinstance(video, str): # video file
                    if audios is not None: # Sample less frames for videos with audio
                        frames, sample_fps = process_video_with_decord(video, self.data_args, 32)
                    else:
                        frames, sample_fps = process_video_with_decord(video, self.data_args, self.data_args.frames_upbound)
                elif isinstance(video, list): # frames
                    total_frames = len(video)
                    max_frames =  self.data_args.frames_upbound  # 最大读取帧数
                    
                    # 根据总帧数决定采样策略
                    if total_frames <= max_frames:
                        # 如果总帧数不超过60，全部加载
                        step = 1
                        selected_indices = list(range(total_frames))
                    else:
                        # 如果总帧数超过60，计算采样间隔
                        step = math.ceil(total_frames / max_frames)
                        # 均匀采样，确保不超过max_frames帧
                        selected_indices = list(range(0, total_frames, step))
                    
                    frames = []
                    frame_numbers = []
                    for i in selected_indices:
                        frame_path = video[i]
                        # 加载图像文件
                        pil_image = Image.open(frame_path)
                        # 转换为numpy数组
                        frame_array = np.array(pil_image)
                        frames.append(frame_array)
                        
                        filename = os.path.basename(frame_path)
                        frame_num = int(filename.split('.')[0])
                        frame_numbers.append(frame_num)
                    
                    if len(frame_numbers) > 1:
                        frame_diffs = []
                        for i in range(1, len(frame_numbers)):
                            frame_diffs.append(frame_numbers[i] - frame_numbers[i-1])
                        sample_fps = sum(frame_diffs) / len(frame_diffs)
                    else:
                        sample_fps = 1.0
                        
                    frames = np.array(frames)
                # 批量处理所有视频帧
                if self.data_args.video_pixel_upbound is not None:
                    frames = np.array([Image.fromarray(frame).resize((self.data_args.video_pixel_upbound, self.data_args.video_pixel_upbound)) for frame in frames])
                c_pixel_values_video, c_video_grid_thw = process_unires_video(frames, self.image_processor, self.data_args.image_grid_pinpoints)
                
                # c_pixel_values_video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
                # c_video_grid_thw = torch.tensor([[frames.shape[0], 1, 1]])
                pixel_values_video.append(c_pixel_values_video)
                video_grid_thw.append(c_video_grid_thw)
                video_sample_fps_list.append(sample_fps)
            
            second_grid_ts = [1 / sample_fps for sample_fps in video_sample_fps_list]
                
            videos_inputs["pixel_values_videos"] = torch.cat(pixel_values_video, dim=0) # seq_len, 3, 384, 384
            videos_inputs["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
            videos_inputs["second_grid_ts"] = torch.tensor(second_grid_ts)
            
        else:
            video_grid_thw = None
        #Added from Sonic-O1
        audio_grid_thw = None
        audios_inputs = {}
    
        if audios is not None:
            audio_time = self.data_args.whisper_audio_time
            whisper_features_list = []
            audio_grid_thw = []
            for j,audio_array in enumerate(audios):
                tmp_sample = audio_array.copy()
                now_len = len(whisper_features_list)
                audio_grid_thw.append(0)
                while len(tmp_sample) > 0:
                    # >30X16000
                    if len(tmp_sample) > 480000:
                        chunk = tmp_sample[:480000]
                        tmp_sample = tmp_sample[480000:]
                        whisper_features = self.audio_processor(chunk,sampling_rate=16000).input_features
                        whisper_features_list.append(torch.Tensor(whisper_features))
                        audio_grid_thw[j]+=1
                    else:
                        # log-Mel
                        whisper_features = self.audio_processor(tmp_sample,sampling_rate=16000).input_features
                        whisper_features_list.append(torch.Tensor(whisper_features))
                        audio_grid_thw[j]+=1
                        tmp_sample = []
                if len(whisper_features_list)-now_len>audio_time: 
                    # print("trunc") 
                    whisper_features_list = whisper_features_list[:now_len+audio_time]
                    audio_grid_thw[j]=audio_time
            
            # FIX: Handle empty audio (no features extracted)
            if len(whisper_features_list) > 0:
                audios_inputs = {
                    "audio_features": torch.cat(whisper_features_list,dim=0),
                    "audio_grid_thw": torch.LongTensor(audio_grid_thw)
                }
            else:
                # No valid audio features - treat as if no audio was provided
                audios_inputs = {}
                audio_grid_thw = None
        

        if not isinstance(text, list):
            text = [text]

        
        if image_grid_thw is not None:
            pool_visual_tokens = self.data_args.vision_spatial_pool_stride ** 2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() * pool_visual_tokens), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if video_grid_thw is not None:
            pool_visual_tokens = self.data_args.vision_spatial_pool_stride ** 2
            index = 0
            for i in range(len(text)):
                while "<|video_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|video_pad|>", "<|placeholder|>" * (video_grid_thw[index].prod() * pool_visual_tokens), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

        if audio_grid_thw is not None:
            audio_chunk_length  = self.data_args.whisper_query_tokens_size # to change
            index = 0
            for i in range(len(text)):
                while "<|audio_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|audio_pad|>", "<|placeholder|>" * (audio_grid_thw[index]*audio_chunk_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|audio_pad|>")
            # print(text)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # labels
        input_ids = text_inputs.input_ids[0]
        label = torch.full(input_ids.size(), IGNORE_INDEX, device=input_ids.device)
        im_start_token =  self.tokenizer("<|im_start|>").input_ids[0]
        im_start_indices = torch.where(input_ids == im_start_token)[0].tolist()
        im_start_indices.extend([-1]) # add end
        for idx in range(len(im_start_indices)):
            if idx == 0 or idx % 2 == 1: # system, user
                continue 
            # plus 1 is "assistant", plus 2 is "\n", plus3 is the start of answer
            # boi, eoi, are trainable <|im_end|>
            label[im_start_indices[idx] + 3: im_start_indices[idx+1]] = input_ids[im_start_indices[idx] + 3: im_start_indices[idx+1]]

        label[-1] = input_ids[-1] # last one should calculate loss
        text_inputs["input_ids"] = text_inputs["input_ids"][0]
        text_inputs["labels"] = label

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs, **audios_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
