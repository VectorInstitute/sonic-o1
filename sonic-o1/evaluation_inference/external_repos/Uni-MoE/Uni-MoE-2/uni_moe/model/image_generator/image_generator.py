import json
import math
import os
import random
import socket

from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import models
from .transport import Sampler, create_transport
from .models.clip.modules import FrozenCLIPEmbedder_Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from .data import read_general


resolution2scale = {
    256: [
        "256x256", "128x512", "144x432", "176x368", "192x336", "224x288",
        "288x224", "336x192", "368x176", "432x144", "512x128"
    ],
    512: ['512 x 512', '1024 x 256', '1008 x 256', '992 x 256', '976 x 256', '960 x 256', '960 x 272',
          '944 x 272', '928 x 272', '912 x 272', '896 x 272', '896 x 288', '880 x 288',
          '864 x 288', '848 x 288', '848 x 304', '832 x 304', '816 x 304', '816 x 320',
          '800 x 320', '784 x 320', '768 x 320', '768 x 336', '752 x 336', '736 x 336',
          '736 x 352', '720 x 352', '704 x 352', '704 x 368', '688 x 368', '672 x 368',
          '672 x 384', '656 x 384', '640 x 384', '640 x 400', '624 x 400', '624 x 416',
          '608 x 416', '592 x 416', '592 x 432', '576 x 432', '576 x 448', '560 x 448',
          '560 x 464', '544 x 464', '544 x 480', '528 x 480', '528 x 496', '512 x 496',
          '496 x 512', '496 x 528', '480 x 528', '480 x 544', '464 x 544',
          '464 x 560', '448 x 560', '448 x 576', '432 x 576', '432 x 592', '416 x 592',
          '416 x 608', '416 x 624', '400 x 624', '400 x 640', '384 x 640', '384 x 656',
          '384 x 672', '368 x 672', '368 x 688', '368 x 704', '352 x 704', '352 x 720',
          '352 x 736', '336 x 736', '336 x 752', '336 x 768', '320 x 768', '320 x 784',
          '320 x 800', '320 x 816', '304 x 816', '304 x 832', '304 x 848', '288 x 848',
          '288 x 864', '288 x 880', '288 x 896', '272 x 896', '272 x 912', '272 x 928',
          '272 x 944', '272 x 960', '256 x 960', '256 x 976', '256 x 992', '256 x 1008',
          '256 x 1024'], 
    768: [  "1536 x 384", "1520 x 384", "1504 x 384", "1488 x 384", "1472 x 384", "1472 x 400",
            "1456 x 400", "1440 x 400", "1424 x 400", "1408 x 400", "1408 x 416", "1392 x 416",
            "1376 x 416", "1360 x 416", "1360 x 432", "1344 x 432", "1328 x 432", "1312 x 432",
            "1312 x 448", "1296 x 448", "1280 x 448", "1264 x 448", "1264 x 464", "1248 x 464",
            "1232 x 464", "1216 x 464", "1216 x 480", "1200 x 480", "1184 x 480", "1184 x 496",
            "1168 x 496", "1152 x 496", "1152 x 512", "1136 x 512", "1120 x 512", "1104 x 512",
            "1104 x 528", "1088 x 528", "1072 x 528", "1072 x 544", "1056 x 544", "1040 x 544",
            "1040 x 560", "1024 x 560", "1024 x 576", "1008 x 576", "992 x 576", "992 x 592",
            "976 x 592", "960 x 592", "960 x 608", "944 x 608", "944 x 624", "928 x 624",
            "912 x 624", "912 x 640", "896 x 640", "896 x 656", "880 x 656", "864 x 656",
            "864 x 672", "848 x 672", "848 x 688", "832 x 688", "832 x 704", "816 x 704",
            "816 x 720", "800 x 720", "800 x 736", "784 x 736", "784 x 752", "768 x 752",
            "768 x 768", "752 x 768", "752 x 784", "736 x 784", "736 x 800", "720 x 800",
            "720 x 816", "704 x 816", "704 x 832", "688 x 832", "688 x 848", "672 x 848",
            "672 x 864", "656 x 864", "656 x 880", "656 x 896", "640 x 896", "640 x 912",
            "624 x 912", "624 x 928", "624 x 944", "608 x 944", "608 x 960", "592 x 960",
            "592 x 976", "592 x 992", "576 x 992", "576 x 1008", "576 x 1024", "560 x 1024",
            "560 x 1040", "544 x 1040", "544 x 1056", "544 x 1072", "528 x 1072", "528 x 1088",
            "528 x 1104", "512 x 1104", "512 x 1120", "512 x 1136", "512 x 1152", "496 x 1152",
            "496 x 1168", "496 x 1184", "480 x 1184", "480 x 1200", "480 x 1216", "464 x 1216",
            "464 x 1232", "464 x 1248", "464 x 1264", "448 x 1264", "448 x 1280", "448 x 1296",
            "448 x 1312", "432 x 1312", "432 x 1328", "432 x 1344", "432 x 1360", "416 x 1360",
            "416 x 1376", "416 x 1392", "416 x 1408", "400 x 1408", "400 x 1424", "400 x 1440",
            "400 x 1456", "400 x 1472", "384 x 1472", "384 x 1488", "384 x 1504", "384 x 1520",
            "384 x 1536"],
    1024: ['1024x1024', '2048x512', '2032x512', '2016x512', '2000x512', '1984x512', '1984x528', '1968x528',
           '1952x528', '1936x528', '1920x528', '1920x544', '1904x544', '1888x544', '1872x544',
           '1872x560', '1856x560', '1840x560', '1824x560', '1808x560', '1808x576', '1792x576',
           '1776x576', '1760x576', '1760x592', '1744x592', '1728x592', '1712x592', '1712x608',
           '1696x608', '1680x608', '1680x624', '1664x624', '1648x624', '1632x624', '1632x640',
           '1616x640', '1600x640', '1584x640', '1584x656', '1568x656', '1552x656', '1552x672',
           '1536x672', '1520x672', '1520x688', '1504x688', '1488x688', '1488x704', '1472x704',
           '1456x704', '1456x720', '1440x720', '1424x720', '1424x736', '1408x736', '1392x736',
           '1392x752', '1376x752', '1360x752', '1360x768', '1344x768', '1328x768', '1328x784',
           '1312x784', '1296x784', '1296x800', '1280x800', '1280x816', '1264x816', '1248x816',
           '1248x832', '1232x832', '1232x848', '1216x848', '1200x848', '1200x864', '1184x864',
           '1184x880', '1168x880', '1168x896', '1152x896', '1136x896', '1136x912', '1120x912',
           '1120x928', '1104x928', '1104x944', '1088x944', '1088x960', '1072x960', 
           '1072x976','1056x976', '1056x992', '1040x992', '1040x1008', '1024x1008', '1008x1024',
           '1008x1040', '992x1040', '992x1056', '976x1056', '976x1072', '960x1072', '960x1088',
           '944x1088', '944x1104', '928x1104', '928x1120', '912x1120', '912x1136', '896x1136',
           '896x1152', '896x1168', '880x1168', '880x1184', '864x1184', '864x1200', '848x1200',
           '848x1216', '848x1232', '832x1232', '832x1248', '816x1248', '816x1264', '816x1280',
           '800x1280', '800x1296', '784x1296', '784x1312', '784x1328', '768x1328', '768x1344',
           '768x1360', '752x1360', '752x1376', '752x1392', '736x1392', '736x1408', '736x1424',
           '720x1424', '720x1440', '720x1456', '704x1456', '704x1472', '704x1488', '688x1488',
           '688x1504', '688x1520', '672x1520', '672x1536', '672x1552', '656x1552', '656x1568',
           '656x1584', '640x1584', '640x1600', '640x1616', '640x1632', '624x1632', '624x1648',
           '624x1664', '624x1680', '608x1680', '608x1696', '608x1712', '592x1712', '592x1728',
           '592x1744', '592x1760', '576x1760', '576x1776', '576x1792', '576x1808', '560x1808',
           '560x1824', '560x1840', '560x1856', '560x1872', '544x1872', '544x1888', '544x1904',
           '544x1920', '528x1920', '528x1936', '528x1952', '528x1968', '528x1984', '512x1984',
           '512x2000', '512x2016', '512x2032', '512x2048']
}

class image_generator(nn.Module):
    def __init__(self):
        super(image_generator, self).__init__()
        # Initialize arguments
        self.num_gpus = 1
        self.precision = "bf16"
        self.proportional_attn = True
        self.scaling_method = "Time-aware"
        self.scaling_watershed = 0.3
        self.debug = False
        self.batch_size = 1
        self.path_type = 'Linear'
        self.prediction = 'velocity'
        self.loss_weight = None
        self.sample_eps = None
        self.train_eps = None
        self.sampling_method = 'euler'
        self.atol = 1e-06
        self.rtol = 0.001
        self.reverse = False
        self.likelihood = False
        # ...other optional arguments...
        self.ckpt = "Uni-MoE-2.0-Image/visual_gen/dit"
        self.sdxl_vae_path = "Uni-MoE-2.0-Image/visual_gen/sdxl-vae"

        self.image_save_path = "examples/assets/visual_gen/generated_images"
        self.num_sampling_steps = 60
        self.seed = 3400
        self.resolution = "768"
        self.time_shifting_factor = 1.0
        self.text_cfg_scale = 4.0
        self.image_cfg_scale = 1.0
        self.ema = True

        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
        self.master_port = str(self.find_free_port())
        self.model, self.vae, self.image_transform, self.train_args = self.load_model(0, self.master_port, self.dtype)

        self.transport = create_transport(self.path_type, self.prediction, self.loss_weight, self.train_eps, self.sample_eps)
        

    def find_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def find_closest_size(self, sizes_list, target_aspect_ratio):
        resolutions = {
            512: 512 * 512,
            768: 768 * 768,
            1024: 1024 * 1024
        }
        min_diff = float('inf')
        closest_aspect_ratio = None
        closest_size = None
        closest_resolution = None

        for size in sizes_list:
            width, height = size.split('x')
            width, height = int(width), int(height)
            area = width * height

            aspect_ratio = width / height
            diff = abs(aspect_ratio - target_aspect_ratio)

            if diff < min_diff:
                min_diff = diff
                closest_aspect_ratio = aspect_ratio
                closest_size = (width, height)
                closest_resolution = min(resolutions, key=lambda x: abs(resolutions[x] - area))

        return closest_aspect_ratio, closest_size, closest_resolution

    def _encode_prompt_with_clip(
        self, text_encoder, tokenizer, prompt, device=None, num_images_per_prompt: int = 1
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
        pooled_prompt_embeds = prompt_embeds[0] #Task Embedding

        return pooled_prompt_embeds

    def encode_prompt(
        self, prompt_batch, text_encoder, tokenizer, clip_tokenizer, clip_text_encoder,
        proportion_empty_prompts, is_train=True, num_images_per_prompt: int = 1, device=None
    ):
        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        clip_pooled_prompt_embeds_list = []
        pooled_prompt_embeds = self._encode_prompt_with_clip( #[3,768]
            text_encoder=clip_text_encoder,
            tokenizer=clip_tokenizer,
            prompt=captions,
            device=device if device is not None else clip_text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        with torch.no_grad():
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            prompt_masks = text_inputs.attention_mask.to(text_encoder.device)

            prompt_embeds = text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_masks,
                output_hidden_states=True,
            ).hidden_states[-2]

        return prompt_embeds, prompt_masks, pooled_prompt_embeds

    def none_or_str(self, value):
        if value == "None":
            return None
        return value

    def parse_transport_args(self, parser):
        group = parser.add_argument_group("Transport arguments")
        group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
        group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
        group.add_argument("--loss-weight", type=self.none_or_str, default=None, choices=[None, "velocity", "likelihood"])
        group.add_argument("--sample-eps", type=float)
        group.add_argument("--train-eps", type=float)

    def parse_ode_args(self, parser):
        group = parser.add_argument_group("ODE arguments")
        group.add_argument(
            "--sampling-method",
            type=str,
            default="euler",
            help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq",
        )
        group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
        group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
        group.add_argument("--reverse", action="store_true")
        group.add_argument("--likelihood", action="store_true")

    def load_model(self, rank, master_port, dtype):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(self.num_gpus)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not dist.is_initialized():
            os.environ["MASTER_PORT"] = str(master_port)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            dist.init_process_group(backend='nccl')

        if fs_init._DATA_PARALLEL_GROUP is None:
            fs_init.initialize_model_parallel(self.num_gpus)

        train_args = torch.load(os.path.join(self.ckpt, "model_args.pth"), weights_only=False)
        prompt_feat_dim = 2048
        prompt_clip_feat_dim = 768
        
        if dist.get_rank() == 0:
            print(f"Creating vae: {train_args.vae}")
        if train_args.vae != "sdxl":
            vae = AutoencoderKL.from_pretrained(
                f"stabilityai/sd-vae-ft-{train_args.vae}", torch_dtype=torch.float32).cuda()
        else:
            vae = AutoencoderKL.from_pretrained( #sdxl
                self.sdxl_vae_path, torch_dtype=torch.float32,
            ).cuda()

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                 inplace=True),
        ])

        if dist.get_rank() == 0:
            print(f"Creating DiT: {train_args.model}")

        model = models.__dict__[train_args.model](
            qk_norm=train_args.qk_norm, #True
            prompt_feat_dim=prompt_feat_dim, #2048
            prompt_clip_feat_dim=prompt_clip_feat_dim, #768
        )

        return model, vae, image_transform, train_args

    def get_next_id(self, directory):
        existing_ids = [int(f.split('_')[0]) for f in os.listdir(directory) if f.split('_')[0].isdigit()]
        return max(existing_ids, default=0) + 1

    def create_blank_image(self, width, height):
        return Image.new("RGB", (width, height), color=(255, 255, 255))

    def forward(self, img_path, tar_path, prompt_feats=None, prompt_pooled_feats=None, input_clip_x=None):
        dtype = self.dtype

        if "-" in self.resolution:
            image_size_list = self.resolution.split('-')
        else:
            image_size_list = [self.resolution]
        image_size_list = [int(num) for num in image_size_list]

        with torch.autocast("cuda", dtype):

            if isinstance(img_path, list):
                img_path = img_path[0]
            if isinstance(tar_path, list):
                tar_path = tar_path[0]

            if img_path:
                inp_img = Image.open(read_general(img_path)).convert("RGB")
            else:
                inp_img = self.create_blank_image(512, 512)  # Default to 512x512 white image
            
            w_gt, h_gt = inp_img.size

            all_resolutions = []
            for res in image_size_list:
                all_resolutions += resolution2scale[res]

            closest_aspect_ratio, closest_size, closest_res = self.find_closest_size(all_resolutions, w_gt/h_gt)
            w, h = closest_size

            latent_w, latent_h = w // 8, h // 8
            z_1 = torch.randn([1, 4, latent_h, latent_w], device=prompt_feats.device).to(dtype)

            inp_img = inp_img.resize((w, h), resample=Image.LANCZOS)
            inp_img = self.image_transform(inp_img).to("cuda") #[3,768,768]

            factor = 0.18215 if self.train_args.vae != "sdxl" else 0.13025
            #golden_input_clip_x = self.clip_Image.encode(inp_img[None]).to(prompt_feats.device) #[1,576,1024] ->clip vision
            input_x = self.vae.encode(inp_img[None]).latent_dist.sample().mul_(factor) #[1,4,96,96] ->VAE Encoder
                
            if input_clip_x is not None:          
                input_clip_x_null = torch.zeros_like(input_clip_x, device=prompt_feats.device).to(dtype) #[1,576,1024]
                input_clip_x = torch.cat((input_clip_x, input_clip_x, input_clip_x_null), dim=0) #[3,576,1024]
            else:
                input_clip_x = torch.load("uni_moe/model/image_generator/embeddings/blank_input_clip_x.pt", map_location="cuda")
            prompt_mask = torch.ones(1, 32)

            prompt_pooled_feats = prompt_pooled_feats [:, :1, :].squeeze(0)
            fallback_image_path = "examples/assets/visual_gen/input_images/white.png"
            try:
                tar_img = Image.open(read_general(tar_path)).convert("RGB")
            except (OSError, UnidentifiedImageError) as e:
                print(f"Warning: Failed to load image at {tar_path}. Reason: {e}. Using fallback image.")
                tar_img = Image.open(fallback_image_path).convert("RGB")
            tar_img = tar_img.resize((w, h), resample=Image.LANCZOS)
            tar_img = self.image_transform(tar_img).to(prompt_feats.device) #[3,768,768]
            target_x = self.vae.encode(tar_img[None]).latent_dist.sample().mul_(factor)
            
            input_x_mb = [input_x.squeeze(0).to()] # [[4,96,96]]
            target_x_mb = [target_x.squeeze(0).to(prompt_feats.device)] #[[4,96,96]]
            input_clip_x_mb = [input_clip_x[:1].to(prompt_feats.device)] #[[1,576,1024]]
            model_kwargs = dict(prompt_feats=prompt_feats.to(prompt_feats.device), prompt_mask=prompt_mask.to(prompt_feats.device), prompt_pooled_feats=prompt_pooled_feats.to(prompt_feats.device)) # [1,8x,2048] [1,8x] [1,768]

            loss_dict = self.transport.training_losses(self.model, input_x_mb, input_clip_x_mb, target_x_mb, model_kwargs)
            loss = loss_dict["loss"].sum()

            #return loss, golden_input_clip_x
            return loss

    def generate_image(self, img_path, prompt=None, prompt_feats=None, prompt_pooled_feats=None, input_clip_x=None, save_path=None):

        if save_path is None:
            save_path = self.image_save_path
        
        if '.png' not in save_path:
            sample_folder_dir = save_path
        else:
            sample_folder_dir = os.path.dirname(save_path)
        rank = 0
        dtype = self.dtype

        if rank == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
            os.makedirs(sample_folder_dir, exist_ok=True)

        if "-" in self.resolution:
            image_size_list = self.resolution.split('-')
        else:
            image_size_list = [self.resolution]
        image_size_list = [int(num) for num in image_size_list]
        print("image resolution center: ", image_size_list)

        next_id = self.get_next_id(sample_folder_dir)

        with torch.autocast("cuda", dtype):
            transport = create_transport(
                self.path_type, self.prediction, self.loss_weight, self.train_eps, self.sample_eps
            )
            sampler = Sampler(transport)
            sample_fn = sampler.sample_ode(
                sampling_method=self.sampling_method,
                num_steps=self.num_sampling_steps,
                atol=self.atol,
                rtol=self.rtol,
                reverse=self.reverse,
                time_shifting_factor=self.time_shifting_factor,
            )

            if int(self.seed) != 0:
                torch.random.manual_seed(int(self.seed))

            if img_path:
                inp_img = Image.open(read_general(img_path)).convert("RGB")
            else:
                inp_img = self.create_blank_image(512, 512)  # Default to 512x512 white image
            
            w_gt, h_gt = inp_img.size #input image size

            all_resolutions = []
            for res in image_size_list:
                all_resolutions += resolution2scale[res]

            closest_aspect_ratio, closest_size, closest_res = self.find_closest_size(all_resolutions, w_gt/h_gt)
            w, h = closest_size

            sample_id = f'{next_id}_{w}x{h}'
            next_id += 1

            res_cat = int(closest_res)
            do_extrapolation = res_cat > 1024

            latent_w, latent_h = w // 8, h // 8
            z_1 = torch.randn([1, 4, latent_h, latent_w], device=prompt_feats.device).to(dtype)
            n = 1
            z = z_1.repeat(n * 3, 1, 1, 1) #[3,4,latent_h,latent_w] ->Random Noice

            inp_img = inp_img.resize((w, h), resample=Image.LANCZOS)
            save_inp_img = inp_img
            inp_img = self.image_transform(inp_img).to(prompt_feats.device) #[3,768,768]

            factor = 0.18215 if self.train_args.vae != "sdxl" else 0.13025
            input_x = self.vae.encode(inp_img[None]).latent_dist.sample().mul_(factor) #[1,4,96,96] ->VAE Encoder
            input_x = torch.cat((input_x, input_x, z_1), dim=0) #[3,4,96,96] ->+Random Noice

            if input_clip_x is not None:          
                input_clip_x_null = torch.zeros_like(input_clip_x, device=prompt_feats.device).to(dtype) #[1,576,1024]
                input_clip_x = torch.cat((input_clip_x, input_clip_x, input_clip_x_null), dim=0) #[3,576,1024]
            else:
                input_clip_x = torch.load("uni_moe/model/image_generator/embeddings/blank_input_clip_x.pt").to(prompt_feats.device)
            
            blank_caps = torch.load("uni_moe/model/image_generator/embeddings/blank_caps.pt").to(prompt_feats.device)
            blank_caps = blank_caps[:, :32]
            prompt_feats = torch.cat((prompt_feats, blank_caps), dim=0)
            prompt_mask = torch.zeros(3, 32)
            prompt_mask[0, :] = 1
            prompt_mask[1, 0] = 1
            prompt_mask[2, 0] = 1
            
            prompt_mask = prompt_mask.to(prompt_feats.device)
            prompt_pooled_feats = prompt_pooled_feats.to(prompt_feats.device)

            model_kwargs = dict(
                input_x=input_x,
                input_clip_x=input_clip_x,
                prompt_feats=prompt_feats,
                prompt_mask=prompt_mask,
                prompt_pooled_feats=prompt_pooled_feats,
                text_cfg_scale=self.text_cfg_scale,
                image_cfg_scale=self.image_cfg_scale,
            )

            if self.proportional_attn:
                model_kwargs["proportional_attn"] = True
                model_kwargs["base_seqlen"] = (res_cat // 16) ** 2
            else:
                model_kwargs["proportional_attn"] = False
                model_kwargs["base_seqlen"] = None

            if do_extrapolation and self.scaling_method == "Time-aware":
                model_kwargs["scale_factor"] = math.sqrt(w * h / res_cat**2)
                model_kwargs["scale_watershed"] = self.scaling_watershed
            else:
                model_kwargs["scale_factor"] = 1.0
                model_kwargs["scale_watershed"] = 1.0

            samples = sample_fn(z, self.model.forward_with_cfg, **model_kwargs)[-1]
            samples = samples[:n]
            
            samples = self.vae.decode(samples / factor).sample
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)

            if prompt is not None:
                # Save samples to disk as individual .png files
                for i, (sample, prompt) in enumerate(zip(samples, prompts_list)):
                    img = to_pil_image(sample.float())
                    if '.png' not in save_path:
                        img.save(f"{save_path}/{sample_id}.png")
                    else:
                        img.save(save_path)
                
                if '.png' not in save_path:
                    save_inp_img.save(f"{save_path}/{sample_id}_input.png")
                with open(f"{save_path}/{sample_id}_prompt.txt", "w") as file:
                    file.write(prompts_list[0])
                    
            else:
                for i, sample in enumerate(samples):
                    img = to_pil_image(sample.float())
                    if '.png' not in save_path:
                        img.save(f"{save_path}/{sample_id}.png")
                    else:
                        img.save(save_path)
                
                if '.png' not in save_path:
                    save_inp_img.save(f"{save_path}/{sample_id}_input.png")

            print(f"Saving .png samples at {sample_folder_dir}")
