from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
# from moviepy.editor import VideoFileClip
from decord import VideoReader, cpu
import numpy as np
from transformers import StoppingCriteria
import torch.distributed as dist

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def process_unires_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    else:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
            
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])
    
    padded_w, padded_h = image_padded.size
    padded_w, padded_h = padded_w // patch_size, padded_h // patch_size

    image_patches = patches
    # 使用批量处理替代列表推导式
    batch_result = processor.preprocess(image_patches, return_tensors="pt")
    image_patches = batch_result["pixel_values"]
    
    image_thw = torch.tensor([[1, padded_h, padded_w]])
    return image_patches, image_thw

def smart_nframes(
    total_frames,
    video_fps,
    fps,
    min_frames,
    max_frames,
) -> int:

    nframes = total_frames / video_fps * fps
    if nframes > total_frames:
        rank0_print(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, 2)
    return nframes


def process_video_with_decord(video_file, data_args, frames_upbound):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    
    fps = data_args.video_fps
    
    nframes = smart_nframes(total_frames, video_fps, fps, 8, frames_upbound)
    idx = torch.linspace(0, total_frame_num - 1, nframes).round().long().tolist()
    try:
        video = vr.get_batch(idx).asnumpy()
    except:
        idx = torch.linspace(4, total_frame_num - 4, nframes).round().long().tolist() # sometimes the first and last frames are corrupted
        video = vr.get_batch(idx).asnumpy()
    vr.seek(0)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    
    return video, sample_fps

def process_unires_video(frames, processor, grid_pinpoints):
    """
    Process video frames with variable resolutions in batch.

    Args:
        frames (np.ndarray): The input video frames in shape (num_frames, height, width, channels).
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A tensor containing the processed video frame patches
            - torch.Tensor: A tensor containing the frame grid dimensions [num_frames, height, width]
    """
    # Convert frames to PIL images for batch processing
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Get patch size and possible resolutions
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    else:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
    
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    
    # Find best resolution based on first frame (assuming all frames have same size)
    best_resolution = select_best_resolution(pil_frames[0].size, possible_resolutions)
    
    # Resize and pad all frames
    frames_padded = [resize_and_pad_image(frame, best_resolution) for frame in pil_frames]
    
    # Get dimensions
    padded_w, padded_h = frames_padded[0].size
    padded_w, padded_h = padded_w // patch_size, padded_h // patch_size
    
    # Divide all frames into patches
    all_patches = []
    for frame in frames_padded:
        patches = divide_to_patches(frame, processor.crop_size["height"])
        all_patches.extend(patches)
    
    # Process all patches in batch
    # 使用批量处理替代列表推导式
    batch_result = processor.preprocess(all_patches, return_tensors="pt")
    processed_patches = batch_result["pixel_values"]
    
    # Reshape to group patches by frame
    patches_per_frame = processed_patches.shape[0] // len(frames)
    processed_patches = processed_patches.view(len(frames), patches_per_frame, *processed_patches.shape[1:])
    
    # Create grid dimensions tensor
    frame_grid_thw = torch.tensor([[len(frames), padded_h, padded_w]])
    
    return processed_patches.reshape(-1, *processed_patches.shape[2:]), frame_grid_thw
    
    # frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    # frame_time = [i/avg_fps for i in frame_idx]

    
    # if data_args.frames_upbound > 0:
    #     if len(frame_idx) > data_args.frames_upbound:
    #         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
    #         frame_idx = uniform_sampled_frames.tolist()
    #         # frame_time = [i/vr.get_avg_fps() for i in frame_idx]
            
    # nframes = len(frame_idx)
    # video = vr.get_batch(frame_idx).asnumpy()
    
    
    # sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    # # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    # # num_frames_to_sample = num_frames = len(frame_idx)
    # # https://github.com/dmlc/decord/issues/208
    # vr.seek(0)
    # return video

# def process_video_with_moviepy(video_file, data_args):
#     # 读取视频文件
#     clip = VideoFileClip(video_file)
    
#     # 获取视频的基本信息
#     total_frame_num = int(clip.fps * clip.duration)  # 视频的总帧数
#     video_fps = data_args.video_fps  # 目标帧率
    
#     # 计算平均帧率和采样间隔
#     avg_fps = round(clip.fps / video_fps)
#     frame_idx = [i for i in range(0, total_frame_num, avg_fps)]  # 按照目标帧率采样的帧索引
#     frame_time = [i / clip.fps for i in frame_idx]  # 对应的时间戳
    
#     # 如果设置了帧数上限，则按需调整帧数
#     if data_args.frames_upbound > 0:
#         if len(frame_idx) > data_args.frames_upbound:
#             uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
#             frame_idx = uniform_sampled_frames.tolist()
#             frame_time = [i / clip.fps for i in frame_idx]
    
#     # 提取视频帧数据
#     frames = []
#     for idx in frame_idx:
#         frame = clip.get_frame(idx / clip.fps)  # 获取特定时间点的帧
#         frames.append(frame)
    
#     # 将帧数据转换为numpy数组
#     video = np.array(frames)
    
#     # 转换为 channel_last 格式（确保是 (num_frames, height, width, channels)）
#     video = np.transpose(video, (0, 2, 3, 1))  # 从 (num_frames, height, width, channels) 格式调整
    
#     # 格式化输出帧时间
#     frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    
#     clip.close()
#     # 返回处理后的视频数据和帧时间
#     return video