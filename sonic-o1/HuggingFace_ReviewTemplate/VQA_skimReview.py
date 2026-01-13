import os
import json
import time
import uuid
import glob
import urllib.parse
import gradio as gr
from huggingface_hub import HfApi, CommitOperationAdd, create_commit, snapshot_download

# --- 1. Configuration and Initial Data Sync ---
PASSCODE = os.environ.get("PASSCODE")
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO_ID = os.environ.get("DATA_REPO_ID")

if DATA_REPO_ID and snapshot_download:
    print(f"🚀 Syncing data from dataset repo: {DATA_REPO_ID}")
    try:
        snapshot_download(
            repo_id=DATA_REPO_ID,
            repo_type="dataset",
            local_dir=".",
            token=HF_TOKEN,
            allow_patterns="vqa/**",
        )
        print("✅ Data sync complete.")
    except Exception as e:
        print(f"❌ Could not sync data from {DATA_REPO_ID}: {e}")
else:
    print("⚠️ Skipping data sync: DATA_REPO_ID secret is not set.")

# --- 2. Session Management ---
SESSIONS_DIR = "sessions"
SESSION_TIMEOUT = 300
os.makedirs(SESSIONS_DIR, exist_ok=True)
_pending_ops = set()

def get_session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def register_session(session_id, repo_path, task, topic):
    try:
        session_data = {
            "file": repo_path, 
            "last_active": time.time(), 
            "task": task,
            "topic": topic
        }
        with open(get_session_file(session_id), 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"✅ Registered session {session_id} for task '{task}' topic '{topic}'")
    except Exception as e:
        print(f"❌ Error registering session {session_id}: {e}")

def update_session_activity(session_id):
    try:
        session_file = get_session_file(session_id)
        if os.path.exists(session_file):
            with open(session_file, 'r') as f: 
                data = json.load(f)
            data["last_active"] = time.time()
            with open(session_file, 'w') as f: 
                json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating session {session_id}: {e}")

def get_active_sessions_info():
    now = time.time()
    active_sessions = {}
    for session_file in glob.glob(os.path.join(SESSIONS_DIR, "*.json")):
        try:
            with open(session_file, 'r') as f: 
                data = json.load(f)
            if now - data.get("last_active", 0) < SESSION_TIMEOUT:
                session_id = os.path.basename(session_file).replace('.json', '')
                active_sessions[session_id] = data
            else:
                os.remove(session_file)
        except Exception:
            pass
    return active_sessions

# --- 3. Core Application Logic ---
def initialize_state(current_state):
    if not isinstance(current_state, dict) or not current_state:
        return {"session_id": str(uuid.uuid4())[:8], "items": [], "idx": 0}
    return current_state

def scan_tasks(base="vqa"):
    """Scan available task directories."""
    if not os.path.isdir(base): 
        return []
    tasks = []
    for d in sorted(os.listdir(base)):
        if os.path.isdir(os.path.join(base, d)) and d.startswith("task"):
            tasks.append(d)
    return tasks

def scan_topics_for_task(task, base="vqa"):
    """Scan available topic JSON files for a given task."""
    if not task:
        return []
    task_dir = os.path.join(base, task)
    if not os.path.isdir(task_dir):
        return []
    topics = []
    for f in sorted(os.listdir(task_dir)):
        if f.endswith(".json"):
            topics.append(f.replace(".json", ""))
    return topics

def update_topic_choices(task):
    """Update topic dropdown when task is selected."""
    topics = scan_topics_for_task(task)
    return gr.update(choices=topics, value=topics[0] if topics else None)

def load_task_topic(task, topic, current_state):
    """Load the selected task and topic JSON file."""
    current_state = initialize_state(current_state)
    print(f"🔵 Loading task '{task}' topic '{topic}' for session {current_state['session_id']}")
    
    if not task or not topic:
        msg = "Please select both a task and topic."
        return gr.update(), msg, *empty_display(), current_state
    
    json_path = os.path.join("vqa", task, f"{topic}.json")
    if not os.path.exists(json_path):
        msg = f"❌ File not found: {json_path}"
        return gr.update(), msg, *empty_display(), current_state
    
    current_state["task"] = task
    current_state["topic"] = topic
    current_state["json_path"] = json_path
    current_state["repo_path"] = path_relative_to_repo(json_path)
    register_session(current_state['session_id'], current_state['repo_path'], task, topic)
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entries = data.get("entries", [])
        
        # For temporal tasks, flatten questions into individual items
        if "temporal" in task:
            flattened_items = []
            for entry_idx, entry in enumerate(entries):
                questions = entry.get("questions", [])
                for q_idx, question in enumerate(questions):
                    # Create a flat item with question at top level
                    flat_item = {
                        "video_id": entry.get("video_id"),
                        "video_number": entry.get("video_number"),
                        "segment": entry.get("segment", {}),
                        "question_id": question.get("question_id"),
                        "question": question.get("question"),
                        "temporal_relation": question.get("temporal_relation"),
                        "anchor_event": question.get("anchor_event"),
                        "target_event": question.get("target_event"),
                        "answer": question.get("answer", {}),
                        "requires_audio": question.get("requires_audio"),
                        "confidence": question.get("confidence"),
                        "abstained": question.get("abstained"),
                        "rationale_model": question.get("rationale_model"),
                        # Keep references for saving back
                        "_entry_idx": entry_idx,
                        "_question_idx": q_idx,
                        # Copy review status if exists
                        "questionCorrect": question.get("questionCorrect"),
                        "review_info": question.get("review_info")
                    }
                    flattened_items.append(flat_item)
            items = flattened_items
        else:
            # For MCQ and summarization, use entries as-is
            items = entries
        
        current_state.update({
            "data": data,
            "entries": entries,  # Keep original structure for writing back
            "items": items,      # Flattened for temporal, original for others
            "idx": 0
        })
        
        status = f"✅ Loaded {len(items)} items from `{current_state['repo_path']}` for session `{current_state['session_id']}`."
        displays = show_current(current_state)
        return gr.update(), status, *displays, current_state
    except Exception as e:
        return gr.update(), f"❌ Error loading JSON: {e}", *empty_display(), current_state

def empty_display():
    """Return empty values for all display components."""
    return (
        "<em>Load a task to begin.</em>",  # video_html
        "<em>No content.</em>",  # content_html
        "—",  # header_md
        "—",  # file_info_md
        gr.update(visible=False),  # segment_btn
        "",  # validation_msg
    )

def show_current(current_state):
    """Display the current entry based on task type."""
    if not current_state or not current_state.get("items"):
        return empty_display()
    
    idx = current_state["idx"]
    items = current_state["items"]
    item = items[idx]
    task = current_state.get("task", "")
    
    # Get video info
    video_id = item.get("video_id", "")
    video_number = item.get("video_number", "")
    
    # Check if reviewed
    is_reviewed = "questionCorrect" in item
    review_status = "✅ Reviewed" if is_reviewed else "⏳ Not reviewed"
    if is_reviewed:
        correct_status = "✅ CORRECT" if item["questionCorrect"] else "❌ INCORRECT"
        review_status += f" - {correct_status}"
    
    # For temporal, add question ID to header
    question_id_display = ""
    if "temporal" in task and item.get("question_id"):
        question_id_display = f"Q#{item['question_id']} | "
    
    # Header
    header = f"Item {idx + 1} / {len(items)} | {question_id_display}Video: {video_number} | {review_status}"
    file_info = f"Task: {task} | Topic: {current_state.get('topic', 'N/A')}"
    
    # Video HTML with segment info
    segment = item.get("segment", {})
    
    # For temporal tasks (now flattened), jump to answer timestamp
    if "temporal" in task:
        answer = item.get("answer", {})
        if answer and answer.get("start_s") is not None:
            answer_start = answer.get("start_s", 0)
            # Create a pseudo-segment with answer timestamp
            video_html = create_video_html(video_id, {"start": answer_start, "end": segment.get("end", 0)}, jump_to_segment=True)
        else:
            video_html = create_video_html(video_id, segment, jump_to_segment=False)
        content_html = format_temporal(item)
        segment_visible = True
    elif "mcq" in task:
        # For MCQ, jump to segment start
        video_html = create_video_html(video_id, segment, jump_to_segment=True)
        content_html = format_mcq(item)
        segment_visible = True
    elif "summarization" in task:
        # For summarization, no auto-jump
        video_html = create_video_html(video_id, segment, jump_to_segment=False)
        content_html = format_summarization(item)
        segment_visible = False
    else:
        video_html = create_video_html(video_id, segment, jump_to_segment=False)
        content_html = "<em>Unknown task type</em>"
        segment_visible = False
    
    return (
        video_html,
        content_html,
        header,
        file_info,
        gr.update(visible=segment_visible),
        ""  # validation_msg
    )

def create_video_html(video_id, segment=None, jump_to_segment=False):
    """Create embedded video HTML with optional segment info and auto-jump."""
    if not video_id:
        return "<em>No video ID available.</em>"
    
    # Determine start time
    start_seconds = 0
    if segment and segment.get("start") is not None:
        start_seconds = int(segment.get("start", 0))
    
    # Add start parameter if jumping to segment or if segment exists
    start_param = f"&start={start_seconds}" if (jump_to_segment or start_seconds > 0) else ""
    
    # Create YouTube embed with start time
    video_html = f'''<div style="position: relative;">
        <iframe id="video-player-{start_seconds}" width="100%" height="400" 
                src="https://www.youtube.com/embed/{video_id}?enablejsapi=1{start_param}" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
        </iframe>
    </div>'''
    
    # Add segment info if available
    if segment and segment.get("start") is not None:
        start_time = format_timestamp(segment.get("start", 0))
        end_time = format_timestamp(segment.get("end", 0))
        video_html += f'''<div style="margin-top: 10px; padding: 10px; background-color: rgba(59, 130, 246, 0.15); 
                          border-radius: 6px; border-left: 3px solid #3b82f6;">
            <strong style="color: #60a5fa;">📍 Segment:</strong> {start_time} - {end_time} ({start_seconds}s - {int(segment.get("end", 0))}s)
        </div>'''
    
    return video_html

def format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def format_summarization(item):
    """Format summarization task display."""
    html = "<div style='background-color: rgba(100, 100, 100, 0.2); padding: 20px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.3);'>"
    
    # Summary Short
    html += "<h3 style='margin-top: 0; color: inherit;'>📝 Summary (Short)</h3>"
    summary_short = item.get("summary_short", [])
    if summary_short:
        html += "<ul style='line-height: 1.8;'>"
        for point in summary_short:
            html += f"<li>{point}</li>"
        html += "</ul>"
    else:
        html += "<em style='opacity: 0.6;'>No short summary available</em>"
    
    html += "<hr style='margin: 20px 0; border-color: rgba(150, 150, 150, 0.3);'>"
    
    # Summary Detailed
    html += "<h3 style='color: inherit;'>📄 Summary (Detailed)</h3>"
    summary_detailed = item.get("summary_detailed", "")
    if summary_detailed:
        html += f"<p style='line-height: 1.8;'>{summary_detailed}</p>"
    else:
        html += "<em style='opacity: 0.6;'>No detailed summary available</em>"
    
    html += "</div>"
    return html

def format_mcq(item):
    """Format MCQ task display."""
    html = "<div style='background-color: rgba(100, 100, 100, 0.2); padding: 20px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.3);'>"
    
    # Question
    html += "<h3 style='margin-top: 0; color: inherit;'>❓ Question</h3>"
    question = item.get("question", "")
    html += f"<p style='font-size: 16px; line-height: 1.6; background-color: rgba(59, 130, 246, 0.15); padding: 15px; border-radius: 6px;'>{question}</p>"
    
    # Options
    html += "<h4 style='margin-top: 20px; color: inherit;'>Options:</h4>"
    options = item.get("options", [])
    answer_index = item.get("answer_index", -1)
    answer_letter = item.get("answer_letter", "")
    
    html += "<div style='margin: 10px 0;'>"
    for i, option in enumerate(options):
        is_correct = (i == answer_index)
        style = "background-color: rgba(34, 197, 94, 0.2); border-left: 3px solid #22c55e;" if is_correct else "background-color: rgba(100, 100, 100, 0.1);"
        marker = "✓ " if is_correct else ""
        html += f"<div style='padding: 10px; margin: 5px 0; border-radius: 4px; {style}'>{marker}{option}</div>"
    
    html += "</div>"
    html += f"<p style='margin-top: 10px;'><strong>Correct Answer:</strong> <span style='color: #22c55e; font-size: 18px;'>{answer_letter}</span></p>"
    
    # Rationale
    html += "<hr style='margin: 20px 0; border-color: rgba(150, 150, 150, 0.3);'>"
    html += "<h4 style='color: inherit;'>💡 Rationale</h4>"
    rationale = item.get("rationale", "")
    if rationale:
        html += f"<p style='line-height: 1.6; background-color: rgba(168, 85, 247, 0.15); padding: 12px; border-radius: 6px;'>{rationale}</p>"
    else:
        html += "<em style='opacity: 0.6;'>No rationale available</em>"
    
    # Evidence tags
    evidence_tags = item.get("evidence_tags", [])
    if evidence_tags:
        html += "<p style='margin-top: 10px;'><strong>Evidence Tags:</strong> "
        for tag in evidence_tags:
            html += f"<span style='background-color: rgba(59, 130, 246, 0.2); padding: 2px 8px; border-radius: 4px; margin-right: 5px;'>{tag}</span>"
        html += "</p>"
    
    # Requires audio
    requires_audio = item.get("requires_audio", False)
    audio_icon = "🔊" if requires_audio else "🔇"
    html += f"<p style='margin-top: 10px;'><strong>Requires Audio:</strong> {audio_icon} {'Yes' if requires_audio else 'No'}</p>"
    
    html += "</div>"
    return html

def format_temporal(item):
    """Format temporal localization task display (flattened structure)."""
    html = "<div style='background-color: rgba(100, 100, 100, 0.2); padding: 20px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.3);'>"
    
    # Get segment info
    segment = item.get("segment", {})
    if segment:
        seg_start = format_timestamp(segment.get("start", 0))
        seg_end = format_timestamp(segment.get("end", 0))
        html += f"<div style='background-color: rgba(168, 85, 247, 0.15); padding: 10px; border-radius: 6px; margin-bottom: 15px; border-left: 3px solid #a855f7;'>"
        html += f"<strong style='color: #c084fc;'>🎬 Context Segment:</strong> {seg_start} - {seg_end}"
        html += "</div>"
    
    # Question (now at top level)
    question = item.get("question", "")
    question_id = item.get("question_id", "")
    
    html += f"<h3 style='margin-top: 0; color: inherit;'>❓ Question {question_id}</h3>"
    html += f"<p style='font-size: 16px; line-height: 1.6; background-color: rgba(59, 130, 246, 0.15); padding: 15px; border-radius: 6px;'>{question}</p>"
    
    # Temporal relation
    html += "<hr style='margin: 20px 0; border-color: rgba(150, 150, 150, 0.3);'>"
    html += "<h4 style='color: inherit;'>⏱️ Temporal Information</h4>"
    
    temporal_relation = item.get("temporal_relation", "")
    anchor_event = item.get("anchor_event", "")
    target_event = item.get("target_event", "")
    
    html += f"<p><strong>Relation:</strong> <span style='background-color: rgba(168, 85, 247, 0.2); padding: 2px 8px; border-radius: 4px;'>{temporal_relation}</span></p>"
    html += f"<p><strong>Anchor Event:</strong> {anchor_event}</p>"
    html += f"<p><strong>Target Event:</strong> {target_event}</p>"
    
    # Answer (timestamp) - HIGHLIGHTED
    html += "<hr style='margin: 20px 0; border-color: rgba(150, 150, 150, 0.3);'>"
    html += "<h4 style='color: inherit;'>✅ Answer (Video jumps here automatically)</h4>"
    answer = item.get("answer", {})
    start_s = answer.get("start_s", 0)
    end_s = answer.get("end_s", 0)
    
    html += f"<div style='background-color: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 6px; border-left: 3px solid #22c55e;'>"
    html += f"<p style='margin: 0; font-size: 18px;'><strong>⏰ Timestamp:</strong> {format_timestamp(start_s)} - {format_timestamp(end_s)}</p>"
    html += f"<p style='margin: 5px 0 0 0; opacity: 0.8; font-size: 13px;'>({start_s:.2f}s - {end_s:.2f}s)</p>"
    html += "</div>"
    
    # Rationale
    rationale = item.get("rationale_model", "")
    if rationale:
        html += "<h4 style='margin-top: 20px; color: inherit;'>💡 Rationale</h4>"
        html += f"<p style='line-height: 1.6; background-color: rgba(168, 85, 247, 0.15); padding: 12px; border-radius: 6px;'>{rationale}</p>"
    
    # Requires audio
    requires_audio = item.get("requires_audio", False)
    audio_icon = "🔊" if requires_audio else "🔇"
    html += f"<p style='margin-top: 10px;'><strong>Requires Audio:</strong> {audio_icon} {'Yes' if requires_audio else 'No'}</p>"
    
    html += "</div>"
    return html

def jump_to_segment(current_state):
    """Reload video with jump to segment start (MCQ) or answer timestamp (Temporal)."""
    if not current_state or not current_state.get("items"):
        return "<em>No segment to jump to</em>"
    
    idx = current_state["idx"]
    item = current_state["items"][idx]
    video_id = item.get("video_id", "")
    segment = item.get("segment", {})
    task = current_state.get("task", "")
    
    # For temporal tasks (now flattened), jump to answer timestamp
    if "temporal" in task:
        answer = item.get("answer", {})
        if answer and answer.get("start_s") is not None:
            answer_start = answer.get("start_s", 0)
            # Create video with answer timestamp
            video_html = create_video_html(video_id, {"start": answer_start, "end": segment.get("end", 0)}, jump_to_segment=True)
            return video_html
    
    # For MCQ and others, jump to segment start
    if not segment or segment.get("start") is None:
        return "<em>No segment information available</em>"
    
    video_html = create_video_html(video_id, segment, jump_to_segment=True)
    return video_html

def move(delta, current_state):
    """Move to next/previous item."""
    if not current_state or not current_state.get("items"):
        return *empty_display(), current_state
    
    update_session_activity(current_state['session_id'])
    new_idx = current_state["idx"] + delta
    current_state["idx"] = max(0, min(len(current_state["items"]) - 1, new_idx))
    
    return *show_current(current_state), current_state

def mark_correctness(is_correct, current_state):
    """Mark the current question as correct or incorrect."""
    if not current_state or not current_state.get("items"):
        return "❌ Load a task first.", *empty_display()[:-1], current_state
    
    update_session_activity(current_state['session_id'])
    idx = current_state["idx"]
    item = current_state["items"][idx]
    
    # Add/update questionCorrect field
    item["questionCorrect"] = is_correct
    
    # Add review metadata
    item["review_info"] = {
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": current_state.get("session_id", "unknown")
    }
    
    # Write back to file
    write_back(current_state)
    _pending_ops.add((current_state["json_path"], current_state["repo_path"]))
    
    status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
    msg = f"✅ Marked as {status} for item {idx + 1}"
    
    # Return updated display
    displays = show_current(current_state)
    return msg, *displays[:-1], current_state

def write_back(current_state):
    """Write the updated data back to JSON file."""
    data = current_state["data"]
    task = current_state.get("task", "")
    
    if "temporal" in task:
        # For temporal, we need to unflatten back to original structure
        entries = current_state["entries"]
        items = current_state["items"]
        
        # Update original entries with review status from flattened items
        for item in items:
            entry_idx = item.get("_entry_idx")
            question_idx = item.get("_question_idx")
            
            if entry_idx is not None and question_idx is not None:
                # Update the specific question in the original entry
                if "questionCorrect" in item:
                    entries[entry_idx]["questions"][question_idx]["questionCorrect"] = item["questionCorrect"]
                if "review_info" in item:
                    entries[entry_idx]["questions"][question_idx]["review_info"] = item["review_info"]
        
        data["entries"] = entries
    else:
        # For MCQ and summarization, items are entries
        data["entries"] = current_state["items"]
    
    with open(current_state["json_path"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_session_status(current_state):
    current_state = initialize_state(current_state)
    sessions = get_active_sessions_info()
    if not sessions:
        return "✅ **No other active sessions.** Safe to push."
    status = f"**Active Sessions ({len(sessions)}):**\n"
    for sid, info in sessions.items():
        marker = "👤 **YOU**" if sid == current_state.get("session_id") else "👥 Other"
        age = int(time.time() - info["last_active"])
        status += f"- {marker}: Session `{sid}` on task **{info.get('task', 'N/A')}** topic **{info.get('topic', 'N/A')}** (active {age}s ago)\n"
    return status

def push_to_dataset(current_state):
    if not (HF_TOKEN and DATA_REPO_ID):
        return "⚠️ Push failed: HF_TOKEN or DATA_REPO_ID secrets are not set."
    if not _pending_ops:
        return "✅ Nothing to push. All changes are already saved."
    try:
        operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=l) for (l, p) in sorted(_pending_ops)]
        create_commit(
            repo_id=DATA_REPO_ID, repo_type="dataset", operations=operations,
            commit_message=f"VQA review update from session {current_state.get('session_id', 'unknown')}",
            token=HF_TOKEN
        )
        num_files = len(_pending_ops)
        _pending_ops.clear()
        return f"✅ **Success!** Pushed {num_files} file(s) to the dataset."
    except Exception as e:
        return f"❌ **Push Failed:** {e}"

def path_relative_to_repo(p): 
    return os.path.relpath(p, os.getcwd())

# --- 5. Gradio User Interface ---
with gr.Blocks(title="VQA Review Tool", theme=gr.themes.Soft()) as demo:
    app_state = gr.State({})
    
    gr.Markdown("# 🎯 VQA Review Tool")
    gr.Markdown("Validate AI-generated questions and answers for video understanding tasks")
    
    with gr.Column() as login_view:
        passcode_input = gr.Textbox(label="Enter Passcode", type="password", placeholder="••••••••")
        login_btn = gr.Button("Unlock")
        login_msg = gr.Markdown("")
    
    with gr.Column(visible=False) as app_view:
        gr.Markdown("### 📋 Instructions")
        gr.Markdown("""
        1. Select a **task type** and **topic**
        2. Watch the video segment (use the timestamp info to jump to the relevant part)
        3. Review the question and answer
        4. Mark as **✅ Correct** or **❌ Incorrect**
        5. Navigate to the next item (auto-saves)
        """)
        
        with gr.Row():
            task_dd = gr.Dropdown(
                choices=scan_tasks(), 
                label="1. Select Task Type",
                value=None
            )
            topic_dd = gr.Dropdown(
                choices=[], 
                label="2. Select Topic",
                value=None
            )
            load_btn = gr.Button("🚀 Load", variant="primary", scale=0)
        
        status_md = gr.Markdown("*Please select a task and topic to begin.*")
        
        header_md = gr.Markdown("—")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Video")
                video_preview = gr.HTML("<em>Video will appear here.</em>")
                segment_jump_btn = gr.Button("⏩ Jump to Segment", visible=False, variant="secondary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Content")
                content_preview = gr.HTML("<em>Content will appear here.</em>")
        
        gr.Markdown("---")
        gr.Markdown("### ✅ Validation")
        
        validation_msg = gr.Markdown("")
        
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", variant="secondary")
            correct_btn = gr.Button("✅ Correct", variant="primary")
            incorrect_btn = gr.Button("❌ Incorrect", variant="stop")
            next_btn = gr.Button("Next ➡️", variant="secondary")
        
        with gr.Accordion("📄 File Information", open=False):
            file_info_md = gr.Markdown("—")
        
        gr.Markdown("---")
        gr.Markdown("### 3. Push Changes to Dataset")
        
        with gr.Row():
            check_btn = gr.Button("🔍 Check Active Sessions")
            push_btn = gr.Button("⬆️ Push to Dataset", variant="primary")
        
        session_status_md = gr.Markdown("")
    
    def unlock_app(code):
        if code == PASSCODE: 
            return gr.update(visible=False), gr.update(visible=True), ""
        return gr.update(), gr.update(), "❌ Incorrect passcode."
    
    login_btn.click(unlock_app, inputs=[passcode_input], outputs=[login_view, app_view, login_msg])
    
    # Update topics when task changes
    task_dd.change(
        update_topic_choices,
        inputs=[task_dd],
        outputs=[topic_dd]
    )
    
    # Load task and topic
    load_btn.click(
        load_task_topic, 
        [task_dd, topic_dd, app_state], 
        [topic_dd, status_md, video_preview, content_preview, header_md, 
         file_info_md, segment_jump_btn, validation_msg, app_state]
    )
    
    # Navigation
    prev_btn.click(
        lambda s: move(-1, s), 
        [app_state], 
        [video_preview, content_preview, header_md, file_info_md, 
         segment_jump_btn, validation_msg, app_state]
    )
    
    next_btn.click(
        lambda s: move(+1, s), 
        [app_state], 
        [video_preview, content_preview, header_md, file_info_md, 
         segment_jump_btn, validation_msg, app_state]
    )
    
    # Validation buttons
    correct_btn.click(
        lambda s: mark_correctness(True, s),
        [app_state],
        [validation_msg, video_preview, content_preview, header_md, 
         file_info_md, segment_jump_btn, app_state]
    )
    
    incorrect_btn.click(
        lambda s: mark_correctness(False, s),
        [app_state],
        [validation_msg, video_preview, content_preview, header_md, 
         file_info_md, segment_jump_btn, app_state]
    )
    
    # Segment jump (reloads video at segment start)
    segment_jump_btn.click(
        jump_to_segment,
        [app_state],
        [video_preview]
    )
    
    # Session and push
    check_btn.click(get_session_status, [app_state], [session_status_md])
    push_btn.click(push_to_dataset, [app_state], [status_md])

if __name__ == "__main__":
    demo.launch()