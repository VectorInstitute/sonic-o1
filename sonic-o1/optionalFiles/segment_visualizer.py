import json
import os
import av
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from pathlib import Path
from PIL import Image

# ================= CONFIGURATION =================
BASE_PATH = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset"
VQA_DIR = os.path.join(BASE_PATH, "vqa")
VIDEO_DIR = os.path.join(BASE_PATH, "dataset", "videos")

TASK_FOLDERS = {
    "summary": "task1_summarization",
    "mcq": "task2_mcq",
    "temporal": "task3_temporal_localization"
}

# Clean color scheme
COLORS = {
    'box_bg': '#f8f9fa',
    'box_border': '#e0e0e0',
    'q_color': '#e67e22',  # Orange for Q
    'a_color': '#27ae60',  # Green for A
    'text_color': '#2c3e50',
    'filmstrip_bg': '#ffffff', 
    'filmstrip_border': '#ffffff', 
    'timestamp_bg': '#ffffff',
    'timestamp_border': '#cccccc',
    'demo_header': '#34495e',
    'demo_label': '#7f8c8d'
}

# ================= DATA LOADING =================
def load_json(path):
    with open(path, 'r') as f: 
        return json.load(f)

def find_valid_example():
    t1_path = os.path.join(VQA_DIR, TASK_FOLDERS['summary'])
    if not os.path.exists(t1_path): 
        return None
    
    json_files = sorted([f for f in os.listdir(t1_path) if f.endswith('.json') and 'backup' not in f])
    
    for json_file in json_files:
        topic = Path(json_file).stem
        t1_data = load_json(os.path.join(t1_path, json_file))
        t2_path = os.path.join(VQA_DIR, TASK_FOLDERS['mcq'], json_file)
        t3_path = os.path.join(VQA_DIR, TASK_FOLDERS['temporal'], json_file)
        
        if not (os.path.exists(t2_path) and os.path.exists(t3_path)): 
            continue
        
        t2_data = load_json(t2_path)
        t3_data = load_json(t3_path)
        t2_dict = {e['video_id']: e for e in t2_data.get('entries', [])}
        t3_dict = {e['video_id']: e for e in t3_data.get('entries', [])}
        
        for t1_entry in t1_data.get('entries', []):
            vid = t1_entry['video_id']
            if vid in t2_dict and vid in t3_dict:
                vid_num = t1_entry.get('video_number', '004')
                vid_path = os.path.join(VIDEO_DIR, topic, f"video_{vid_num}.mp4")
                if os.path.exists(vid_path):
                    return {
                        'topic': topic, 
                        'video_path': vid_path,
                        'summary': t1_entry, 
                        'mcq': t2_dict[vid], 
                        'temporal': t3_dict[vid]
                    }
    return None

# ================= HELPERS =================
def extract_frames(video_path, start_t, end_t, num_frames=10):
    frames = []
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        if end_t <= start_t: 
            end_t = start_t + 10
        timestamps = np.linspace(start_t, end_t, num_frames)
        
        for ts in timestamps:
            container.seek(int(ts / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                img = frame.to_image()
                frames.append(img)
                break
        container.close()
    except Exception as e:
        print(f"Error extracting frames: {e}")
        for _ in range(num_frames):
            frames.append(Image.new('RGB', (300, 200), color='lightgray'))
    
    return frames

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def format_demographics(demographics):
    if not demographics:
        return ""
    demo_parts = []
    for demo in demographics:
        parts = []
        if 'race' in demo: parts.append(f"Race: {demo['race']}")
        if 'gender' in demo: parts.append(f"Gender: {demo['gender']}")
        if 'age' in demo: parts.append(f"Age: {demo['age']}")
        if 'language' in demo: parts.append(f"Language: {demo['language']}")
        demo_parts.append(" | ".join(parts))
    return " • ".join(demo_parts)

def draw_filmstrip(ax, frames, bottom_y, avail_width, start_time, end_time, demographics=None):
    """Draw a filmstrip with CORRECT aspect ratio relative to the subplot"""
    n_frames = len(frames)
    
    # Get content aspect ratio
    if frames:
        im_w, im_h = frames[0].size
        video_aspect = im_w / im_h
    else:
        video_aspect = 16/9

    # 1. Determine Width of a single frame in Axes Coordinates
    # We want tight packing, so spacing is negligible
    frame_spacing = 0.001 
    total_spacing = frame_spacing * (n_frames - 1)
    
    strip_left = (1.0 - avail_width) / 2
    frame_w_axes = (avail_width - total_spacing) / n_frames
    
    # 2. Determine Height of that frame in Axes Coordinates
    # This is the crucial fix. 
    # We need the aspect ratio of the AXES (the subplot box itself) to convert width->height
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    
    # The physical aspect ratio of the subplot (width / height)
    ax_aspect_ratio = (bbox.width * fig_w) / (bbox.height * fig_h)
    
    # Mathematical derivation:
    # frame_physical_width / frame_physical_height = video_aspect
    # (frame_w_axes * ax_physical_width) / (frame_h_axes * ax_physical_height) = video_aspect
    # frame_h_axes = frame_w_axes * (ax_physical_width / ax_physical_height) / video_aspect
    frame_h_axes = frame_w_axes * ax_aspect_ratio / video_aspect

    # Draw frames
    for i, frame in enumerate(frames):
        x_pos = strip_left + i * (frame_w_axes + frame_spacing)
        
        # Create inset axes exactly sized to the aspect ratio
        frame_ax = ax.inset_axes([x_pos, bottom_y, frame_w_axes, frame_h_axes], transform=ax.transAxes)
        frame_ax.imshow(frame) # aspect='equal' is default behavior if box is correct ratio
        frame_ax.axis('off')

    # 3. Text Labels (Timestamps & Demographics)
    # Position them relative to the bottom of the filmstrip
    label_y = bottom_y - 0.05
    
    # Left timestamp
    ax.text(
        strip_left, label_y,
        start_time,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['timestamp_bg'], 
                  edgecolor=COLORS['timestamp_border'], linewidth=0.5)
    )
    
    # Right timestamp
    ax.text(
        strip_left + avail_width, label_y,
        end_time,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['timestamp_bg'], 
                  edgecolor=COLORS['timestamp_border'], linewidth=0.5)
    )
    
    # Center demographics
    if demographics:
        demo_text = format_demographics(demographics)
        if demo_text:
            ax.text(
                0.5, label_y,
                f"Demographics:  {demo_text}",
                transform=ax.transAxes,
                fontsize=8,
                ha='center',
                va='top',
                color=COLORS['demo_label'],
                style='italic'
            )

def draw_qa_block(ax, question, answer, frames, start_time, end_time, demographics=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Background
    bg_box = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01",
        transform=ax.transAxes,
        facecolor=COLORS['box_bg'],
        edgecolor=COLORS['box_border'],
        linewidth=1.5,
        zorder=0
    )
    ax.add_patch(bg_box)
    
    text_x = 0.04
    text_width = 110
    
    # Q
    q_label = "Q:"
    q_text = textwrap.fill(question, width=text_width)
    
    ax.text(text_x, 0.92, q_label, transform=ax.transAxes, fontsize=12, fontweight='bold', color=COLORS['q_color'], va='top')
    ax.text(text_x + 0.025, 0.92, q_text, transform=ax.transAxes, fontsize=12, color=COLORS['text_color'], va='top', linespacing=1.3)
    
    # A - Calculation to prevent overlap
    q_lines = len(q_text.split('\n'))
    line_height = 0.06 # Increased line height calculation for safety
    a_start_y = 0.92 - (q_lines * line_height) - 0.02
    
    a_label = "A:"
    a_text = textwrap.fill(answer, width=text_width)
    
    ax.text(text_x, a_start_y, a_label, transform=ax.transAxes, fontsize=12, fontweight='bold', color=COLORS['a_color'], va='top')
    ax.text(text_x + 0.025, a_start_y, a_text, transform=ax.transAxes, fontsize=12, color=COLORS['text_color'], va='top', linespacing=1.3)
    
    # Filmstrip Positioning
    # Anchor the filmstrip near the bottom, but ensure it clears the text
    # With a 3-row layout (figsize 16x9), the filmstrip frames need to be roughly 0.28 height
    # to be 16:9. We anchor them at 0.12 to leave room for demographics below.
    filmstrip_bottom_y = 0.12
    
    draw_filmstrip(ax, frames, filmstrip_bottom_y, 0.92, start_time, end_time, demographics)

# ================= MAIN =================
def create_dashboard(data):
    if not data: return
    
    container = av.open(data['video_path'])
    duration = float(container.streams.video[0].duration * container.streams.video[0].time_base)
    container.close()
    
    # Task 1
    summary_raw = data['summary'].get('summary_short', [])
    if isinstance(summary_raw, list): summary_raw = " ".join(summary_raw)
    q1 = "What is the main topic of this video?"
    a1 = summary_raw[:280] + "..." if len(summary_raw) > 280 else summary_raw
    frames1 = extract_frames(data['video_path'], 0, duration, 10)
    
    # Task 2
    mcq = data['mcq']
    if 'questions' in mcq and isinstance(mcq['questions'], list): mcq = mcq['questions'][0]
    q2 = mcq.get('question', 'N/A')
    answer_letter = mcq.get('answer', '') or mcq.get('answer_letter', '')
    options = mcq.get('options', {})
    if isinstance(options, list): options = {chr(65+i): opt for i, opt in enumerate(options)}
    
    raw_answer_text = options.get(answer_letter.strip('()'), answer_letter)
    # Clean answer
    clean_answer_text = raw_answer_text.strip()
    prefixes = [f"({answer_letter})", f"{answer_letter}.", f"{answer_letter} "]
    for p in prefixes:
        if clean_answer_text.startswith(p):
            clean_answer_text = clean_answer_text[len(p):].strip()
            break
    a2 = f"({answer_letter}) {clean_answer_text}"
    
    mcq_start = duration * 0.25
    mcq_end = duration * 0.65
    frames2 = extract_frames(data['video_path'], mcq_start, mcq_end, 10)
    
    # Task 3
    temporal = data['temporal']
    if 'questions' in temporal and temporal['questions']: temporal = temporal['questions'][0]
    q3 = temporal.get('question', 'N/A')
    temp_start = temporal.get('start', 0)
    temp_end = temporal.get('end', 10)
    if temp_start == 0 and temp_end <= 10:
        temp_start = duration * 0.1
        temp_end = duration * 0.3
    a3 = f"Target Segment: {int(temp_start)}s - {int(temp_end)}s"
    frames3 = extract_frames(data['video_path'], temp_start, temp_end, 10)
    
    # Layout
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs = fig.add_gridspec(3, 1, hspace=0.05, top=0.98, bottom=0.02, left=0.01, right=0.99)
    
    ax1 = fig.add_subplot(gs[0, 0])
    draw_qa_block(ax1, q1, a1, frames1, format_timestamp(0), format_timestamp(duration), data['summary'].get('demographics'))
    
    ax2 = fig.add_subplot(gs[1, 0])
    draw_qa_block(ax2, q2, a2, frames2, format_timestamp(mcq_start), format_timestamp(mcq_end), data['mcq'].get('demographics'))
    
    ax3 = fig.add_subplot(gs[2, 0])
    draw_qa_block(ax3, q3, a3, frames3, format_timestamp(temp_start), format_timestamp(temp_end), data['temporal'].get('demographics'))
    
    output_path = "aesthetic_dashboard_clean.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    example = find_valid_example()
    if example:
        print(f"Generating dashboard for: {example['topic']}")
        create_dashboard(example)
    else:
        print("No valid data found.")