import os
import io
import time
from dataclasses import dataclass, field
import json
import zipfile
from typing import Dict, Optional, Sequence, List, Any, Union, Tuple
import random
import numpy as np
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset
from uni_moe.model import *
from uni_moe.qwen_vl_utils import process_mm_info
from PIL import Image
import pandas as pd
from uni_moe.train.training_utils import rank0_pprint

IGNORE_INDEX = -100


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def extract_image_from_zip(zip_path, image_to_extract):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(image_to_extract) as image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
    return image

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    ep_size: int = field(default=1)
    
    token_drop: bool = field(default=False)
    frames_upbound: int = field(default=64)
    
    tune_image_generator: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    visual_gen_embeddings_folder: Optional[str] = field(default=None)
    image_folder: Optional[str] = field(default="")
    video_folder: Optional[str] = field(default="")
    audio_folder: Optional[str] = field(default="")
    code_folder: Optional[str] = field(default="")

    aux_balance_weight: float = field(default=None)
    
    stable_mode: bool = False
     


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=32768,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    freeze_prefix: Optional[List[str]] = field(default=None)
    group_by_modality_length: bool = field(default=False)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="sdpa", metadata={"help": "Use transformers attention implementation."})
    per_device_train_batch_size: int = 1
    l_aux_weight: float = field(default=0.001)
    balanced_mega_batch_size: int = field(default=None)
    lora_enable: bool = field(default=False)
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    
    
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 augment_config_path=None,
                 ):
        super(LazySupervisedDataset, self).__init__()
        if data_path.endswith('.json'):
            list_data_dict = json.load(open(data_path, "r"))
        elif os.path.isdir(data_path):
            list_data_dict = []
            sorted_file_names = sorted(os.listdir(data_path))
            for file_name in sorted_file_names:
                if file_name.endswith('.json'):
                    list_data_dict.extend(json.load(open(os.path.join(data_path, file_name), "r")))
                elif file_name.endswith('.parquet'):
                    df = pd.read_parquet(os.path.join(data_path, file_name))
                    df_list = df.to_dict(orient='records')
                    list_data_dict.extend(df_list)
            
        for record in list_data_dict:
            for key, value in record.items():
                if isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                if isinstance(value, list) and len(value) == 0:
                    record[key] = None
                    
        self.data_path = data_path
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.mono = True
        self.sample_rate = 16000
        self.meta_data_type = "llava"
        self.stable_mode = data_args.stable_mode
    

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths_type(self):
        length_list = []
        data_type = ["videos", "audios", "images", "tts"] 
        # use 3 bits to represent the modality
        # only text is 000, only image is 001, video is 010, audios is 100
        # video and audios is 110, video and image is 011, audios and image is 101
        # video, audios and image is 111
        print("---------------sampling with textual len-------------")
        for sample in self.list_data_dict:
            # cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if sample.get("textual_length"):
                cur_len = sample["textual_length"]
            else:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_type = 0
            if sample.get("codes"):
                cur_type += 8
            if sample.get("videos"):
                cur_type += 4
            if sample.get("audios"):
                cur_type += 2
            if sample.get("images"):
                cur_type += 1
            length_list.append((cur_len, cur_type))
        return length_list

    def modality_type(self,sample):
        cur_type = 0
        if sample.get("codes"):
            cur_type += 8
        if sample.get("videos"):
            cur_type += 4
        if sample.get("audios"):
            cur_type += 2
        if sample.get("images"):
            cur_type += 1
        return cur_type


    def meta_form(self, meta_type, sources):
        if "llava" in meta_type:
            # llava data: {"image":[],"audio":[],"conversations":[{"from":,"value":},{"from":,"value":}]}
            messages = []
            images = audios = videos = None
            images_idx = audios_idx = videos_idx = 0
            if sources.get("images"):  ## Important Remember to change.....
                images = sources["images"]
                if type(images) == str: images = [images]
                images = [os.path.join(self.data_args.image_folder, image) for image in images]
                images_len = len(images)
            if sources.get("audios"): 
                audios = sources["audios"]
                if type(audios) == str: audios = [audios]
                if sources.get("videos") and ".mp4" in audios[0]: # special for video with audioA
                    audios = [os.path.join(self.data_args.video_folder, audio) for audio in audios]
                else:
                    audios = [os.path.join(self.data_args.audio_folder, audio) for audio in audios]
                audios_len = len(audios)
            if sources.get("videos"): 
                videos = sources["videos"]
                if type(videos) == str: videos = [videos]
                assert isinstance(videos[0], str)
                if videos[0][0] != "[":
                    videos = [os.path.join(self.data_args.video_folder, video) for video in videos]
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
                if "<image>" in tag_count: assert images is not None, "Image tag found but no images provided!"
                if "<video>" in tag_count: assert videos is not None, "Video tag found but no videos provided!"
                if "<audio>" in tag_count: assert audios is not None, "Audio tag found but no audios provided!"
                
                if conv["from"] == "human":
                    conv_dic = {'role': 'user', "content":[{"type": "text", "text": text}]}
                elif conv["from"] == "system":
                    conv_dic = {'role': 'system', "content":[{"type": "text", "text": text}]}
                else:
                    conv_dic = {'role': 'assistant', "content":[{"type": "text", "text": text}]}
                    
                while images is not None and "<image>" in tag_count:
                    assert images_len > images_idx
                    if sources.get("resized_height") and sources.get("resized_width"):
                        conv_dic["content"].append({"type": "image", "image": images[images_idx], "resized_height": sources["resized_height"], "resized_width": sources["resized_width"]})
                    else:
                        conv_dic["content"].append({"type": "image", "image": images[images_idx]})
                    tag_count = tag_count.replace("<image>", "", 1)
                    images_idx+=1
                while audios is not None and "<audio>" in tag_count:
                    assert audios_len > audios_idx
                    conv_dic["content"].append({"type": "audio", "audio": audios[audios_idx]})
                    tag_count = tag_count.replace("<audio>", "", 1)
                    audios_idx+=1
                while videos is not None and "<video>" in tag_count:
                    assert videos_len > videos_idx
                    conv_dic["content"].append({"type": "video", "video": videos[videos_idx]})
                    tag_count = tag_count.replace("<video>", "", 1)
                    videos_idx+=1
                messages.append(conv_dic)   
            messages = [messages] # batch size = 1

            return messages
        else:
            return sources

    def deal_audio_out(self,sources,inputs):
        assert "codes" in sources, "Try to tune speech out with out codes part!"
        # tmp_ids = inputs["input_ids"]
        all_input_ids = inputs["input_ids"]
        all_labels = inputs["labels"]

        split_id = self.data_args.speech_split_token_id
        split_id_indices = torch.where(all_input_ids == split_id)[0]
        if len(split_id_indices)>0:
            new_input_ids = torch.cat([all_input_ids[:split_id_indices[0]]]+[all_input_ids[split_id_indices[i]+1:split_id_indices[i+1]] for i in range(len(split_id_indices)-1)]+[all_input_ids[split_id_indices[-1]+1:]],dim=0)
            new_labels = torch.cat([all_labels[:split_id_indices[0]]]+[all_labels[split_id_indices[i]+1:split_id_indices[i+1]] for i in range(len(split_id_indices)-1)]+[all_labels[split_id_indices[-1]+1:]],dim=0)
        else:
            new_input_ids = all_input_ids
            new_labels = all_labels
        
        assert new_input_ids.shape[-1] + len(split_id_indices) == all_input_ids.shape[-1]
        split_id_indices -= torch.arange(len(split_id_indices))
        speech_num = len(split_id_indices)+len(torch.where(all_input_ids == self.data_args.speech_start_token_id)[0])
        # print(inputs["input_ids"],split_id_indices)

        inputs["input_ids"] = new_input_ids
        inputs["labels"] = new_labels

        inputs["speech_splits"] = split_id_indices
            # print(inputs["input_ids"].shape,inputs["labels"].shape)

        # codes label of the speech
        # self.data_args.code_folder
        if type(sources["codes"]) == str and sources["codes"][0]!="[":
            with open(os.path.join(self.data_args.code_folder, sources["codes"]),"r") as f:
                dict_codes = json.load(f)
        else:
            real_codes = sources["codes"]
            if sources["codes"][0]=="[":
                real_codes = json.loads(sources["codes"])
            # dict_codes = sources["codes"]
            dict_codes = []
            for code_li in real_codes:
                lis = []
                for code in code_li:
                    with open(os.path.join(self.data_args.code_folder, code),"r") as f:
                        dict_code = json.load(f)
                        lis.append(dict_code)
                dict_codes.append(lis)
        
        short_codes = []
        maxl = 0
        for whole_speech in dict_codes:
            for short_speech in whole_speech:
                short_codes.append(torch.LongTensor(short_speech))
                if len(short_speech)>maxl:
                    maxl = len(short_speech)
        #         assert len(short_speech)<9000,"length too big!!!"
        # assert len(short_codes)<111,"bs too big!!!"
        assert len(short_codes)*maxl<10500, "all too long!"
        assert maxl<1650, "code too long!"
        assert speech_num == len(short_codes),"no match!!!"
        inputs["codes"] = short_codes
        # print("codes",inputs["codes"])
        return inputs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """Get item with robust error handling and fallback mechanisms."""
        if self.stable_mode:
            return self._get_item_with_fallback(i)
        else:
            return self._get_item(i)
    
    def _get_item_with_fallback(self, i: int) -> Dict[str, torch.Tensor]:
        """Get item with fallback to alternative samples when errors occur."""
        # Constants for retry logic
        NUM_BASE_RETRIES = 3
        TEXTUAL_LENGTH_THRESHOLD = 2500
        PROBE_LOG_INTERVAL = 20000
        RETRY_SLEEP_TIME = 1
        
        # First attempt: try the requested sample directly
        sample = self._try_get_item_with_retries(i, NUM_BASE_RETRIES, RETRY_SLEEP_TIME)
        if sample is not None:
            return sample
        
        # Second attempt: try alternative samples with similar characteristics
        ddlen = len(self.list_data_dict)
        cur_data = self.list_data_dict[i]
        cur_modality_type = self.modality_type(cur_data)
        # cur_textual_length = self._get_textual_length(cur_data)
        cur_textual_length = cur_data["textual_length"] if "textual_length" in cur_data else sum(len(conv['value'].split()) for conv in cur_data['conversations'])
        
        sample = self._try_alternative_samples(
            i, cur_data, cur_modality_type, cur_textual_length, 
            NUM_BASE_RETRIES * 100000, TEXTUAL_LENGTH_THRESHOLD, PROBE_LOG_INTERVAL, ddlen
        )
        if sample is not None:
            return sample
        
        # Final attempt: try the original sample one more time
        try:
            return self._get_item(i)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch sample {i} after all retry attempts. Last error: {e}") from e
    
    def _try_get_item_with_retries(self, index: int, max_retries: int, sleep_time: int) -> Optional[Dict[str, torch.Tensor]]:
        """Try to get item with specified number of retries."""
        for attempt_idx in range(max_retries):
            try:
                return self._get_item(index)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample: {index}, id: {self.list_data_dict[index]['id']}, modality: {self.modality_type(self.list_data_dict[index])}. Exception: {e}")
                if attempt_idx < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(sleep_time)
        return None
    
    def _try_alternative_samples(self, original_idx: int, cur_data: dict, cur_modality_type: int, 
                               cur_textual_length: int, max_retries: int, length_threshold: int, 
                               probe_log_interval: int, ddlen: int) -> Optional[Dict[str, torch.Tensor]]:
        """Try alternative samples with similar characteristics."""
        probe_idx = (original_idx + 1) % ddlen
        
        for attempt_idx in range(max_retries):
            try:
                next_index, probe_idx = self._find_similar_sample(
                    original_idx, attempt_idx, cur_modality_type, cur_textual_length,
                    length_threshold, probe_idx, probe_log_interval, ddlen
                )
                
                sample = self._get_item(next_index)
                # next_textual_length = self._get_textual_length(self.list_data_dict[next_index])
                next_textual_length = self.list_data_dict[next_index]["textual_length"]
                print(f"[!!Try other #{attempt_idx}] Success to fetch sample {next_index}. "
                      f"Original length: {cur_textual_length}, Original ID: {self.list_data_dict[original_idx]['id']}, Original Modality: {cur_modality_type}, New length: {next_textual_length}, New Modality: {self.modality_type(self.list_data_dict[next_index])}, New ID: {self.list_data_dict[next_index]['id']}")
                return sample
                
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception: {e}")
                continue
        
        return None
    
    def _find_similar_sample(self, original_idx: int, attempt_idx: int, target_modality_type: int,
                           target_textual_length: int, length_threshold: int, probe_idx: int,
                           probe_log_interval: int, ddlen: int) -> Tuple[int, int]:
        """Find a sample with similar characteristics to the target.
        
        Returns:
            tuple: (next_index, updated_probe_idx)
        """
        next_index = (original_idx + 1 + attempt_idx) % ddlen
        try_time = 0
        max_tries = ddlen  # Prevent infinite loops
        
        print(f"Target Modality Type: {target_modality_type}, Modality Type Origin: {self.modality_type(self.list_data_dict[original_idx])}")
        
        while try_time < max_tries:
            next_data = self.list_data_dict[next_index]
            next_textual_length = next_data["textual_length"]
            next_modality_type = self.modality_type(next_data)
            
            # Check if this sample meets our criteria
            length_diff = abs(target_textual_length - next_textual_length)
            if length_diff < length_threshold and target_modality_type == next_modality_type:
                break
            
            # Log probing progress periodically
            if probe_idx % probe_log_interval == 0:
                print(f"[Probe] probing {next_index}.")
            
            next_index = probe_idx
            probe_idx = (probe_idx + 1) % ddlen
            try_time += 1
        
        return next_index, probe_idx
    
    def _get_item(self, i) -> Dict[str, torch.Tensor]:
    # def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # select the data
        sources = self.list_data_dict[i]

        messages = self.meta_form(self.meta_data_type,sources)
        # messages, golden_task_embedding, golden_caption_embedding, golden_visual_embedding, img_path, tar_path = self.meta_form(self.meta_data_type,sources)

        Qwen_processor = self.data_args.processor
        
        prompt = Qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, no=True
        )
        
        for pi,pr in enumerate(prompt):
            prompt[pi] = prompt[pi].replace("<image>","<|vision_start|><|image_pad|><|vision_end|>").replace("<audio>","<|audio_start|><|audio_pad|><|audio_end|>").replace("<video>","<|vision_start|><|video_pad|><|vision_end|>")

        # print(prompt) # prompt check

        # preprocess modality infomations to modality list 
        image_inputs, video_inputs, audio_inputs = process_mm_info(messages)

        inputs = Qwen_processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            padding=True,
            return_tensors="pt",
        )

        # visual_gen
        if sources.get("golden_caption_embedding"):
            golden_task_embedding = sources.get("golden_task_embedding", None)
            golden_caption_embedding = sources.get("golden_caption_embedding", None)
            golden_visual_embedding = sources.get("golden_visual_embedding", None)
            img_path = sources["images"][0]
            tar_path = sources["image_gen"]
            inputs["golden_visual_embedding"] = golden_visual_embedding
            inputs["img_path"] = img_path
            inputs["tar_path"] = tar_path
        
        modality_id = self.modality_type(sources)

        
        inputs["modality_id"] = modality_id
    
        return inputs


def qwen_pad(seq_list,batch_first=True,padding_value=0,padding_side="right"):
    # seq_list: [torch.tensor([]),torch.tensor([]),...]
    max_len = max([t.shape[-1] for t in seq_list])

    if padding_side=="right":
        return torch.stack([torch.cat([t,torch.full((max_len-t.shape[-1],),padding_value)],dim=0) for t in seq_list]),max_len
    elif padding_side=="left":
        return torch.stack([torch.cat([torch.full((max_len-t.shape[-1],),padding_value),t],dim=0) for t in seq_list]),max_len
    else:
        raise ValueError("padding side can be ether left or right")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    processor: Any
    aux_balance_weight: float
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        set_max = False


        input_ids, ids_max_len = qwen_pad(
            input_ids,
            batch_first=True,
            padding_value = self.processor.tokenizer.pad_token_id,
            padding_side = self.processor.tokenizer.padding_side,)
        labels, _ = qwen_pad(labels,
                        batch_first=True,
                        padding_value=IGNORE_INDEX,
                        padding_side = self.processor.tokenizer.padding_side,)
        # print(input_ids)
        input_ids = input_ids[:, :self.processor.tokenizer.model_max_length]
        labels = labels[:, :self.processor.tokenizer.model_max_length]
        
        batch = {}
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] = input_ids.ne(self.processor.tokenizer.pad_token_id)


        # audio_features / pixel_values / pixel_values_videos / image_grid_thw / video_grid_thw / audio_grid_thw
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat([instance["pixel_values"] for instance in instances],dim=0)
            batch["image_grid_thw"] = torch.cat([instance["image_grid_thw"] for instance in instances],dim=0)
        if "pixel_values_videos" in instances[0]:
            batch["pixel_values_videos"] = torch.cat([instance["pixel_values_videos"] for instance in instances],dim=0)
            batch["video_grid_thw"] = torch.cat([instance["video_grid_thw"] for instance in instances],dim=0)
            batch["second_per_grid_ts"] = torch.cat([instance["second_grid_ts"] for instance in instances],dim=0)
        if "audio_features" in instances[0]:
            batch["audio_features"] = torch.cat([instance["audio_features"] for instance in instances],dim=0)
            batch["audio_grid_thw"] = torch.cat([instance["audio_grid_thw"] for instance in instances],dim=0)
            # print("batch shape",batch["audio_features"].shape,batch["audio_grid_thw"])

        # visual_gen
        #print(f"dataloader instances{instances}")
        if "tar_path" in instances[0]:
            # s1 s2
            batch["golden_visual_embedding"] = [instance["golden_visual_embedding"] for instance in instances]
            batch["img_path"] = [instance["img_path"] for instance in instances]
            batch["tar_path"] = [instance["tar_path"] for instance in instances]
            
        if "codes" in instances[0]:
            codes= [instance["codes"] for instance in instances]
            new_codes = []
            for code in codes:
                new_codes += code
            codes = torch.nn.utils.rnn.pad_sequence(new_codes,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX,
                                                    padding_side = "right")
            batch["codes"] = codes

        if "speech_splits" in instances[0]:
            # make sure the speech splits sit on the exact place when padding side is different
            ids_lens = [instance["input_ids"].shape[-1] for instance in instances]
            if self.processor.tokenizer.padding_side == "left":
                speech_splits = [instance["speech_splits"]+(ids_max_len-ids_len) for instance,ids_len in zip(instances,ids_lens)]
            else:
                speech_splits = [instance["speech_splits"] for instance in instances]
            speech_splits = torch.nn.utils.rnn.pad_sequence(speech_splits,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX,
                                                 padding_side = "right") if speech_splits[0] is not None else None
            batch["speech_splits"] = speech_splits
            
        if self.aux_balance_weight is not None:
            aux_balance_weight = torch.ones_like(input_ids)
            aux_balance_weight[labels != IGNORE_INDEX] = self.aux_balance_weight
            batch["aux_balance_weight"] = aux_balance_weight  

        return batch


def make_supervised_data_module(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(data_path=data_args.data_path,
                                        data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(processor=data_args.processor, aux_balance_weight=data_args.aux_balance_weight, data_args=data_args)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
