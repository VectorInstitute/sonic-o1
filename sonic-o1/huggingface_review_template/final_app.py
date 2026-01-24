import contextlib
import json
import os
import time
import uuid

import gradio as gr
from huggingface_hub import CommitOperationAdd, create_commit, snapshot_download


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
            allow_patterns=["vqa/**", "captions/**"],
        )
        print("✅ Data sync complete.")
    except Exception as e:
        print(f"❌ Could not sync data from {DATA_REPO_ID}: {e}")
else:
    print("⚠️ Skipping data sync: DATA_REPO_ID secret is not set.")

# --- 2. Session Management ---
SESSIONS_DIR = "sessions"
SESSION_TIMEOUT = 2000
os.makedirs(SESSIONS_DIR, exist_ok=True)

_pending_ops = {}
_active_sessions = {}


def get_session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def register_session(session_id, repo_path, task, topic):
    _active_sessions[session_id] = {
        "file": repo_path,
        "last_active": time.time(),
        "task": task,
        "topic": topic,
    }
    print(f"✅ Registered session {session_id} for task '{task}' topic '{topic}'")


def update_session_activity(session_id):
    if session_id in _active_sessions:
        _active_sessions[session_id]["last_active"] = time.time()


def get_active_sessions_info():
    now = time.time()
    active_sessions = {}

    expired = [
        sid
        for sid, data in _active_sessions.items()
        if now - data.get("last_active", 0) >= SESSION_TIMEOUT
    ]
    for sid in expired:
        del _active_sessions[sid]

    for sid, data in _active_sessions.items():
        if now - data.get("last_active", 0) < SESSION_TIMEOUT:
            active_sessions[sid] = data

    return active_sessions


# --- 3. Core Application Logic ---
def initialize_state(current_state):
    if not isinstance(current_state, dict) or not current_state:
        return {"session_id": str(uuid.uuid4())[:8], "items": [], "idx": 0}
    return current_state


def scan_tasks(base="vqa"):
    if not os.path.isdir(base):
        return []
    tasks = []
    for d in sorted(os.listdir(base)):
        if os.path.isdir(os.path.join(base, d)) and d.startswith("task"):
            tasks.append(d)
    return tasks


def scan_topics_for_task(task, base="vqa"):
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
    topics = scan_topics_for_task(task)
    return gr.update(choices=topics, value=topics[0] if topics else None)


def load_caption_for_item(topic, video_number):
    try:
        if not video_number:
            return ""

        caption_file = os.path.join("captions", topic, f"caption_{video_number}.srt")

        if not os.path.exists(caption_file):
            return f"⚠️ Caption file not found: caption_{video_number}.srt"

        with open(caption_file, "r", encoding="utf-8") as f:
            caption_text = f.read()

        if not caption_text.strip():
            return "⚠️ Caption file is empty."

        return caption_text

    except Exception as e:
        return f"❌ Error loading caption: {str(e)}"


def load_task_topic(task, topic, current_state):
    current_state = initialize_state(current_state)
    print(
        f"🔵 Loading task '{task}' topic '{topic}' for session {current_state['session_id']}"
    )

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
    register_session(
        current_state["session_id"], current_state["repo_path"], task, topic
    )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])

        if "temporal" in task:
            flattened_items = []
            for entry_idx, entry in enumerate(entries):
                questions = entry.get("questions", [])
                for q_idx, question in enumerate(questions):
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
                        "_entry_idx": entry_idx,
                        "_question_idx": q_idx,
                        "demographics": entry.get("demographics", []),
                        "demographics_explanation": entry.get(
                            "demographics_explanation", ""
                        ),
                    }
                    flattened_items.append(flat_item)
            items = flattened_items
        else:
            items = entries

        current_state.update(
            {"data": data, "entries": entries, "items": items, "idx": 0}
        )

        status = f"✅ Loaded {len(items)} items from `{current_state['repo_path']}` for session `{current_state['session_id']}`."
        displays = show_current(current_state)
        return gr.update(), status, *displays, current_state
    except Exception as e:
        import traceback

        traceback.print_exc()
        return (
            gr.update(),
            f"❌ Error loading JSON: {e}",
            *empty_display(),
            current_state,
        )


def empty_display():
    return (
        "<em>Load a task to begin.</em>",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "[]",
        "—",
        "—",
        gr.update(visible=False),
        "",
    )


def show_current(current_state):
    if not current_state or not current_state.get("items"):
        return empty_display()

    idx = current_state["idx"]
    items = current_state["items"]
    item = items[idx]
    task = current_state.get("task", "")
    topic = current_state.get("topic", "")

    video_id = item.get("video_id", "")
    video_number = item.get("video_number", "")

    header = f"Item {idx + 1} / {len(items)} | Video: {video_number}"
    file_info = f"Task: {task} | Topic: {topic}"

    caption_text = load_caption_for_item(topic, video_number)

    segment = item.get("segment", {})

    fields = [""] * 17

    if "summarization" in task:
        video_html = create_video_html(video_id, segment, jump_to_segment=False)
        summary_vis = gr.update(visible=True)
        mcq_vis = gr.update(visible=False)
        temporal_vis = gr.update(visible=False)
        segment_visible = False

        fields[0] = "\n".join(item.get("summary_short", []))
        fields[1] = item.get("summary_detailed", "")

    elif "mcq" in task:
        video_html = create_video_html(video_id, segment, jump_to_segment=True)
        summary_vis = gr.update(visible=False)
        mcq_vis = gr.update(visible=True)
        temporal_vis = gr.update(visible=False)
        segment_visible = True

        fields[2] = item.get("question", "")
        fields[3] = "\n".join(item.get("options", []))
        fields[4] = item.get("answer_letter", "")
        fields[5] = item.get("rationale", "")
        fields[6] = str(item.get("requires_audio", False))
        fields[7] = item.get("demographics_explanation", "")

    elif "temporal" in task:
        answer = item.get("answer", {})
        if answer and answer.get("start_s") is not None:
            answer_start = answer.get("start_s", 0)
            video_html = create_video_html(
                video_id,
                {"start": answer_start, "end": segment.get("end", 0)},
                jump_to_segment=True,
            )
        else:
            video_html = create_video_html(video_id, segment, jump_to_segment=False)
        summary_vis = gr.update(visible=False)
        mcq_vis = gr.update(visible=False)
        temporal_vis = gr.update(visible=True)
        segment_visible = True

        fields[8] = item.get("question", "")
        fields[9] = item.get("temporal_relation", "")
        fields[10] = item.get("anchor_event", "")
        fields[11] = item.get("target_event", "")
        fields[12] = str(answer.get("start_s", ""))
        fields[13] = str(answer.get("end_s", ""))
        fields[14] = item.get("rationale_model", "")
        fields[15] = str(item.get("requires_audio", False))
        fields[16] = item.get("demographics_explanation", "")

    else:
        video_html = create_video_html(video_id, segment, jump_to_segment=False)
        summary_vis = gr.update(visible=False)
        mcq_vis = gr.update(visible=False)
        temporal_vis = gr.update(visible=False)
        segment_visible = False

    demographics_json = json.dumps(item.get("demographics", []), indent=2)

    return (
        video_html,
        caption_text,
        summary_vis,
        mcq_vis,
        temporal_vis,
        *fields,
        demographics_json,
        header,
        file_info,
        gr.update(visible=segment_visible),
        "",
    )


def create_video_html(video_id, segment=None, jump_to_segment=False):
    if not video_id:
        return "<em>No video ID available.</em>"

    start_seconds = 0
    if segment and segment.get("start") is not None:
        start_seconds = int(segment.get("start", 0))

    start_param = (
        f"&start={start_seconds}" if (jump_to_segment or start_seconds > 0) else ""
    )

    video_html = f"""<div style="position: relative;">
        <iframe id="video-player-{start_seconds}" width="100%" height="400"
                src="https://www.youtube.com/embed/{video_id}?enablejsapi=1{start_param}"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
        </iframe>
    </div>"""

    if segment and segment.get("start") is not None:
        start_time = format_timestamp(segment.get("start", 0))
        end_time = format_timestamp(segment.get("end", 0))
        video_html += f"""<div style="margin-top: 10px; padding: 10px; background-color: rgba(59, 130, 246, 0.15);
                          border-radius: 6px; border-left: 3px solid #3b82f6;">
            <strong style="color: #60a5fa;">📍 Segment:</strong> {start_time} - {end_time}
        </div>"""

    return video_html


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def jump_to_segment(current_state):
    if not current_state or not current_state.get("items"):
        return "<em>No segment to jump to</em>"

    idx = current_state["idx"]
    item = current_state["items"][idx]
    video_id = item.get("video_id", "")
    segment = item.get("segment", {})
    task = current_state.get("task", "")

    if "temporal" in task:
        answer = item.get("answer", {})
        if answer and answer.get("start_s") is not None:
            answer_start = answer.get("start_s", 0)
            return create_video_html(
                video_id,
                {"start": answer_start, "end": segment.get("end", 0)},
                jump_to_segment=True,
            )

    if not segment or segment.get("start") is None:
        return "<em>No segment information available</em>"

    return create_video_html(video_id, segment, jump_to_segment=True)


# NEW FUNCTION: Jump to specific question number
def jump_to_question(question_num, current_state, demographics_json, *edit_fields):
    """Jump to a specific question number (auto-saves current question first)."""
    if not current_state or not current_state.get("items"):
        return "❌ Load a task first.", *empty_display(), current_state

    try:
        target_idx = int(question_num) - 1  # Convert to 0-based index

        if target_idx < 0 or target_idx >= len(current_state["items"]):
            return (
                f"❌ Invalid question number. Please enter a number between 1 and {len(current_state['items'])}.",
                *show_current(current_state),
                current_state,
            )

        # Auto-save current question before jumping
        save_edits(current_state, demographics_json, *edit_fields, silent=True)

        # Jump to target question
        current_state["idx"] = target_idx
        update_session_activity(current_state["session_id"])

        return (
            f"✅ Jumped to question {question_num}",
            *show_current(current_state),
            current_state,
        )

    except ValueError:
        return (
            "❌ Please enter a valid number.",
            *show_current(current_state),
            current_state,
        )


def save_edits(current_state, demographics_json, *edit_fields, silent=False):
    if not current_state or not current_state.get("items"):
        return "❌ Load a task first.", current_state

    update_session_activity(current_state["session_id"])

    idx = current_state["idx"]
    item = current_state["items"][idx]
    task = current_state.get("task", "")

    try:
        demographics = json.loads(demographics_json)
        item["demographics"] = demographics
    except json.JSONDecodeError:
        if not silent:
            return "❌ Invalid demographics JSON", current_state

    if "summarization" in task:
        summary_short_text = edit_fields[0]
        item["summary_short"] = [
            line.strip() for line in summary_short_text.split("\n") if line.strip()
        ]
        item["summary_detailed"] = edit_fields[1]

    elif "mcq" in task:
        item["question"] = edit_fields[2]
        options_text = edit_fields[3]
        item["options"] = [
            line.strip() for line in options_text.split("\n") if line.strip()
        ]
        item["answer_letter"] = edit_fields[4]
        if edit_fields[4] and item["options"]:
            with contextlib.suppress(BaseException):
                item["answer_index"] = ord(edit_fields[4].upper()) - ord("A")
        item["rationale"] = edit_fields[5]
        item["requires_audio"] = edit_fields[6].lower() in ["true", "1", "yes"]
        item["demographics_explanation"] = edit_fields[7]

    elif "temporal" in task:
        item["question"] = edit_fields[8]
        item["temporal_relation"] = edit_fields[9]
        item["anchor_event"] = edit_fields[10]
        item["target_event"] = edit_fields[11]
        try:
            item["answer"]["start_s"] = float(edit_fields[12]) if edit_fields[12] else 0
            item["answer"]["end_s"] = float(edit_fields[13]) if edit_fields[13] else 0
        except:
            pass
        item["rationale_model"] = edit_fields[14]
        item["requires_audio"] = edit_fields[15].lower() in ["true", "1", "yes"]
        item["demographics_explanation"] = edit_fields[16]

    write_back(current_state)

    session_id = current_state["session_id"]
    if session_id not in _pending_ops:
        _pending_ops[session_id] = set()
    _pending_ops[session_id].add(
        (current_state["json_path"], current_state["repo_path"])
    )

    if silent:
        return "", current_state
    return f"✅ Changes saved for item {idx + 1}", current_state


def move(delta, current_state, demographics_json, *edit_fields):
    if not current_state or not current_state.get("items"):
        return *empty_display(), current_state

    save_edits(current_state, demographics_json, *edit_fields, silent=True)

    update_session_activity(current_state["session_id"])
    new_idx = current_state["idx"] + delta
    current_state["idx"] = max(0, min(len(current_state["items"]) - 1, new_idx))

    return *show_current(current_state), current_state


def write_back(current_state):
    data = current_state["data"]
    task = current_state.get("task", "")

    if "temporal" in task:
        entries = current_state["entries"]
        items = current_state["items"]

        for item in items:
            entry_idx = item.get("_entry_idx")
            question_idx = item.get("_question_idx")

            if entry_idx is not None and question_idx is not None:
                question = entries[entry_idx]["questions"][question_idx]
                question["question"] = item.get("question", "")
                question["temporal_relation"] = item.get("temporal_relation", "")
                question["anchor_event"] = item.get("anchor_event", "")
                question["target_event"] = item.get("target_event", "")
                question["answer"] = item.get("answer", {})
                question["rationale_model"] = item.get("rationale_model", "")
                question["requires_audio"] = item.get("requires_audio", False)

                entries[entry_idx]["demographics"] = item.get("demographics", [])
                entries[entry_idx]["demographics_explanation"] = item.get(
                    "demographics_explanation", ""
                )

        data["entries"] = entries
    else:
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

    session_id = current_state.get("session_id")
    if not _pending_ops.get(session_id):
        return "✅ Nothing to push. All changes are already saved."

    try:
        operations = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=l)
            for (l, p) in sorted(_pending_ops[session_id])
        ]
        create_commit(
            repo_id=DATA_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=f"VQA review update from session {session_id}",
            token=HF_TOKEN,
        )
        num_files = len(_pending_ops[session_id])
        _pending_ops[session_id].clear()
        return f"✅ **Success!** Pushed {num_files} file(s) to the dataset."
    except Exception as e:
        return f"❌ **Push Failed:** {e}"


def path_relative_to_repo(p):
    return os.path.relpath(p, os.getcwd())


# --- 5. Gradio User Interface ---
with gr.Blocks(title="VQA Review & Edit Tool", theme=gr.themes.Soft()) as demo:
    app_state = gr.State({})

    gr.Markdown("# 🎯 VQA Review & Edit Tool")
    gr.Markdown("Edit and validate AI-generated video understanding annotations")

    with gr.Column() as login_view:
        passcode_input = gr.Textbox(
            label="Enter Passcode", type="password", placeholder="••••••••"
        )
        login_btn = gr.Button("Unlock")
        login_msg = gr.Markdown("")

    with gr.Column(visible=False) as app_view:
        gr.Markdown("### 📋 Instructions")
        gr.Markdown("""
        1. Select a **task type** and **topic**
        2. Watch the video and read the caption
        3. **Edit the fields below** to correct any errors
        4. Changes **auto-save** when you navigate
        """)

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=scan_tasks(), label="1. Select Task Type", value=None
            )
            topic_dd = gr.Dropdown(choices=[], label="2. Select Topic", value=None)
            load_btn = gr.Button("🚀 Load", variant="primary", scale=0)

        status_md = gr.Markdown("*Please select a task and topic to begin.*")
        header_md = gr.Markdown("—")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Video")
                video_preview = gr.HTML("<em>Video will appear here.</em>")
                segment_jump_btn = gr.Button(
                    "⏩ Jump to Segment", visible=False, variant="secondary"
                )

                gr.Markdown("### 📄 Caption")
                caption_display = gr.Textbox(
                    label="",
                    lines=10,
                    max_lines=15,
                    value="",
                    interactive=False,
                    show_label=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("### ✏️ Edit Data")

                # SUMMARIZATION FIELDS
                with gr.Column(visible=False) as summary_container:
                    summary_short_field = gr.Textbox(
                        label="Summary (Short) - one bullet point per line",
                        lines=6,
                        placeholder="• Point 1\n• Point 2\n• Point 3",
                    )
                    summary_detailed_field = gr.Textbox(
                        label="Summary (Detailed)",
                        lines=10,
                        placeholder="Enter detailed summary paragraph",
                    )

                # MCQ FIELDS
                with gr.Column(visible=False) as mcq_container:
                    mcq_question_field = gr.Textbox(
                        label="Question", lines=3, placeholder="Enter the question"
                    )
                    mcq_options_field = gr.Textbox(
                        label="Options (one per line)",
                        lines=5,
                        placeholder="(A) First option\n(B) Second option\n(C) Third option",
                    )
                    mcq_answer_field = gr.Textbox(
                        label="Correct Answer Letter",
                        placeholder="e.g., B",
                        max_lines=1,
                    )
                    mcq_rationale_field = gr.Textbox(
                        label="Rationale (Explanation)",
                        lines=4,
                        placeholder="Explain why this is the correct answer",
                    )
                    mcq_audio_field = gr.Textbox(
                        label="Requires Audio (true/false)",
                        placeholder="true or false",
                        max_lines=1,
                    )
                    mcq_demo_explain_field = gr.Textbox(
                        label="Demographics Explanation",
                        lines=3,
                        placeholder="Explain the demographics visible in the video",
                    )

                # TEMPORAL FIELDS
                with gr.Column(visible=False) as temporal_container:
                    temporal_question_field = gr.Textbox(
                        label="Question",
                        lines=3,
                        placeholder="Enter the temporal question",
                    )
                    temporal_relation_field = gr.Textbox(
                        label="Temporal Relation",
                        placeholder="e.g., after, during, next, once_finished",
                        max_lines=1,
                    )
                    temporal_anchor_field = gr.Textbox(
                        label="Anchor Event",
                        lines=2,
                        placeholder="Describe the anchor event",
                    )
                    temporal_target_field = gr.Textbox(
                        label="Target Event",
                        lines=2,
                        placeholder="Describe the target event",
                    )
                    with gr.Row():
                        temporal_start_field = gr.Textbox(
                            label="Answer Start (seconds)",
                            placeholder="e.g., 34.5",
                            scale=1,
                        )
                        temporal_end_field = gr.Textbox(
                            label="Answer End (seconds)",
                            placeholder="e.g., 36.8",
                            scale=1,
                        )
                    temporal_rationale_field = gr.Textbox(
                        label="Rationale (Explanation)",
                        lines=4,
                        placeholder="Explain the temporal relationship and timestamps",
                    )
                    temporal_audio_field = gr.Textbox(
                        label="Requires Audio (true/false)",
                        placeholder="true or false",
                        max_lines=1,
                    )
                    temporal_demo_explain_field = gr.Textbox(
                        label="Demographics Explanation",
                        lines=3,
                        placeholder="Explain the demographics visible in the video",
                    )

                # DEMOGRAPHICS (common to all tasks)
                gr.Markdown("### 👥 Demographics (JSON)")
                demographics_field = gr.Code(
                    label="Edit demographics as JSON array",
                    language="json",
                    lines=8,
                    value="[]",
                )

        save_msg = gr.Markdown("")

        # NEW: Go to Question feature
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", variant="secondary", scale=1)
            manual_save_btn = gr.Button("💾 Save", variant="primary", scale=1)
            next_btn = gr.Button("Next ➡️", variant="secondary", scale=1)

            with gr.Column(scale=1), gr.Row():
                goto_input = gr.Number(
                    label="Go to Question #", precision=0, minimum=1, scale=2
                )
                goto_btn = gr.Button("🎯 Jump", variant="secondary", scale=1)

        with gr.Accordion("📄 File Information", open=False):
            file_info_md = gr.Markdown("—")

        gr.Markdown("---")
        gr.Markdown("### 3. Push Changes to Dataset")

        with gr.Row():
            check_btn = gr.Button("🔍 Check Active Sessions")
            push_btn = gr.Button("⬆️ Push to Dataset", variant="primary")

        session_status_md = gr.Markdown("")

    # Group all edit fields in order
    all_edit_fields = [
        summary_short_field,
        summary_detailed_field,
        mcq_question_field,
        mcq_options_field,
        mcq_answer_field,
        mcq_rationale_field,
        mcq_audio_field,
        mcq_demo_explain_field,
        temporal_question_field,
        temporal_relation_field,
        temporal_anchor_field,
        temporal_target_field,
        temporal_start_field,
        temporal_end_field,
        temporal_rationale_field,
        temporal_audio_field,
        temporal_demo_explain_field,
    ]

    def unlock_app(code):
        if code == PASSCODE:
            return gr.update(visible=False), gr.update(visible=True), ""
        return gr.update(), gr.update(), "❌ Incorrect passcode."

    login_btn.click(
        unlock_app, inputs=[passcode_input], outputs=[login_view, app_view, login_msg]
    )

    task_dd.change(update_topic_choices, inputs=[task_dd], outputs=[topic_dd])

    load_btn.click(
        load_task_topic,
        [task_dd, topic_dd, app_state],
        [
            topic_dd,
            status_md,
            video_preview,
            caption_display,
            summary_container,
            mcq_container,
            temporal_container,
            *all_edit_fields,
            demographics_field,
            header_md,
            file_info_md,
            segment_jump_btn,
            save_msg,
            app_state,
        ],
    )

    prev_btn.click(
        lambda s, d, *fields: move(-1, s, d, *fields),
        [app_state, demographics_field, *all_edit_fields],
        [
            video_preview,
            caption_display,
            summary_container,
            mcq_container,
            temporal_container,
            *all_edit_fields,
            demographics_field,
            header_md,
            file_info_md,
            segment_jump_btn,
            save_msg,
            app_state,
        ],
    )

    next_btn.click(
        lambda s, d, *fields: move(+1, s, d, *fields),
        [app_state, demographics_field, *all_edit_fields],
        [
            video_preview,
            caption_display,
            summary_container,
            mcq_container,
            temporal_container,
            *all_edit_fields,
            demographics_field,
            header_md,
            file_info_md,
            segment_jump_btn,
            save_msg,
            app_state,
        ],
    )

    # NEW: Jump to question handler
    goto_btn.click(
        jump_to_question,
        [goto_input, app_state, demographics_field, *all_edit_fields],
        [
            save_msg,
            video_preview,
            caption_display,
            summary_container,
            mcq_container,
            temporal_container,
            *all_edit_fields,
            demographics_field,
            header_md,
            file_info_md,
            segment_jump_btn,
            save_msg,
            app_state,
        ],
    )

    manual_save_btn.click(
        lambda s, d, *fields: save_edits(s, d, *fields, silent=False),
        [app_state, demographics_field, *all_edit_fields],
        [save_msg, app_state],
    )

    segment_jump_btn.click(jump_to_segment, [app_state], [video_preview])

    check_btn.click(get_session_status, [app_state], [session_status_md])
    push_btn.click(push_to_dataset, [app_state], [status_md])

if __name__ == "__main__":
    demo.launch()
