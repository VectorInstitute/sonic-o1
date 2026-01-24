import glob
import json
import os
import time
import urllib.parse
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
            allow_patterns="captions/**",
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


def register_session(session_id, repo_path, topic):
    try:
        session_data = {"file": repo_path, "last_active": time.time(), "topic": topic}
        with open(get_session_file(session_id), "w") as f:
            json.dump(session_data, f, indent=2)
        print(f"✅ Registered session {session_id} for topic '{topic}'")
    except Exception as e:
        print(f"❌ Error registering session {session_id}: {e}")


def update_session_activity(session_id):
    try:
        session_file = get_session_file(session_id)
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                data = json.load(f)
            data["last_active"] = time.time()
            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating session {session_id}: {e}")


def get_active_sessions_info():
    now = time.time()
    active_sessions = {}
    for session_file in glob.glob(os.path.join(SESSIONS_DIR, "*.json")):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            if now - data.get("last_active", 0) < SESSION_TIMEOUT:
                session_id = os.path.basename(session_file).replace(".json", "")
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


def load_topic(topic, current_state):
    current_state = initialize_state(current_state)
    print(f"🔵 Loading topic '{topic}' for session {current_state['session_id']}")

    jp = choose_json_for_topic(topic)
    if not jp:
        msg = f"No metadata_enhanced.json file found for topic '{topic}'."
        return (
            gr.update(),
            msg,
            "<em>No video.</em>",
            "<em>No caption.</em>",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            current_state,
        )

    current_state["topic"] = topic
    current_state["json_path"] = jp
    current_state["repo_path"] = path_relative_to_repo(jp)
    register_session(current_state["session_id"], current_state["repo_path"], topic)

    try:
        with open(jp, "r", encoding="utf-8") as f:
            root = json.load(f)
        path, items = best_array_in_json(root)
        current_state.update({"root": root, "items": items, "path": path, "idx": 0})
        status = f"✅ Loaded {len(items)} items from `{current_state['repo_path']}` for session `{current_state['session_id']}`."
        (
            video_html,
            caption_html,
            header,
            topic_file,
            demographics_html,
            race_txt,
            gender_txt,
            age_txt,
            lang_txt,
        ) = show_current(current_state)
        return (
            gr.update(choices=scan_topics(), value=topic),
            status,
            video_html,
            caption_html,
            header,
            topic_file,
            demographics_html,
            race_txt,
            gender_txt,
            age_txt,
            lang_txt,
            "",
            current_state,
        )
    except Exception as e:
        return (
            gr.update(),
            f"❌ Error loading JSON: {e}",
            "<em>Error</em>",
            "<em>Error</em>",
            "—",
            "—",
            "",
            "",
            "",
            "",
            "",
            "",
            current_state,
        )


def show_current(current_state):
    if not current_state or not current_state.get("items"):
        return (
            "<em>Load a topic to begin.</em>",
            "<em>No caption.</em>",
            "—",
            "—",
            "",
            "",
            "",
            "",
            "",
        )

    idx = current_state["idx"]
    items = current_state["items"]
    item = items[idx]

    id_field = detect_id_field(items)
    item_id = str(item.get(id_field, idx))
    url = item.get("url", "")

    video_html = to_embed_html(url)
    caption_html = load_caption_for_item(current_state, item)

    # Check if this item has been reviewed
    is_reviewed = "demographics_detailed_reviewed" in item
    review_status = "✅ Reviewed" if is_reviewed else "⏳ Not reviewed"

    header = f"Item {idx + 1} / {len(items)} | ID: {item_id} | {review_status}"
    topic_file = f"Topic: {current_state.get('topic', 'N/A')}"

    # Get demographics data
    demographics_html, race_text, gender_text, age_text, lang_text = (
        format_demographics(item)
    )

    return (
        video_html,
        caption_html,
        header,
        topic_file,
        demographics_html,
        race_text,
        gender_text,
        age_text,
        lang_text,
    )


def format_demographics(item):
    """Formats the demographics data for display and editing - DARK MODE COMPATIBLE."""
    demographics_detailed = item.get("demographics_detailed", {})
    demographics_confidence = item.get("demographics_confidence", {})
    demographics_annotation = item.get("demographics_annotation", {})

    # Check if there's a human review - if so, use that for the text boxes
    demographics_reviewed = item.get("demographics_detailed_reviewed", {})
    use_reviewed = bool(demographics_reviewed)

    # Extract current values (prefer reviewed version if exists)
    source_data = demographics_reviewed if use_reviewed else demographics_detailed
    races = source_data.get("race", [])
    genders = source_data.get("gender", [])
    ages = source_data.get("age", [])
    languages = source_data.get("language", [])

    # Build HTML display with dark mode support
    html = "<div style='background-color: rgba(100, 100, 100, 0.2); padding: 20px; border-radius: 8px; border: 1px solid rgba(150, 150, 150, 0.3);'>"

    if use_reviewed:
        html += "<h3 style='margin-top: 0; color: inherit;'>✅ Human-Reviewed Demographics</h3>"
        review_info = item.get("demographics_review_info", {})
        if review_info:
            html += "<p style='font-size: 13px; opacity: 0.8;'>"
            html += f"<strong>Reviewed at:</strong> {review_info.get('reviewed_at', 'N/A')} | "
            html += (
                f"<strong>Session:</strong> {review_info.get('session_id', 'N/A')}</p>"
            )
    else:
        html += "<h3 style='margin-top: 0; color: inherit;'>🤖 AI-Generated Demographics</h3>"

    # Show original AI annotation metadata
    if demographics_annotation:
        html += "<p style='font-size: 13px; opacity: 0.8;'>"
        html += f"<strong>AI Model:</strong> {demographics_annotation.get('model', 'N/A')} | "
        html += f"<strong>Individuals:</strong> {demographics_annotation.get('individuals_count', 'N/A')} | "
        html += f"<strong>AI Annotated:</strong> {demographics_annotation.get('annotated_at', 'N/A')}</p>"

    # Explanation
    if demographics_annotation.get("explanation"):
        html += "<div style='background-color: rgba(59, 130, 246, 0.15); padding: 12px; border-radius: 6px; margin: 10px 0; border-left: 3px solid #3b82f6;'>"
        html += f"<strong style='color: #60a5fa;'>AI Explanation:</strong><br><em style='font-size: 13px; opacity: 0.9;'>{demographics_annotation['explanation']}</em></div>"

    html += "<hr style='margin: 15px 0; border-color: rgba(150, 150, 150, 0.3);'>"

    # Display current demographics with confidence (always show AI version for reference)
    html += "<h4 style='margin-bottom: 10px; color: inherit;'>Original AI Demographics:</h4>"
    html += "<table style='width: 100%; border-collapse: collapse;'>"

    for category, values in [
        ("Race", demographics_detailed.get("race", [])),
        ("Gender", demographics_detailed.get("gender", [])),
        ("Age", demographics_detailed.get("age", [])),
        ("Language", demographics_detailed.get("language", [])),
    ]:
        html += "<tr style='border-bottom: 1px solid rgba(150, 150, 150, 0.3);'>"
        html += f"<td style='padding: 8px; font-weight: bold; width: 120px; color: inherit;'>{category}:</td>"
        html += "<td style='padding: 8px;'>"
        if values:
            value_list = []
            for val in values:
                conf = demographics_confidence.get(category.lower(), {}).get(val, 0)
                value_list.append(
                    f"<span style='background-color: rgba(34, 197, 94, 0.2); padding: 2px 8px; border-radius: 4px; margin-right: 5px; border: 1px solid rgba(34, 197, 94, 0.4); color: inherit;'>{val} <span style='opacity: 0.7; font-size: 11px;'>({conf:.1%})</span></span>"
                )
            html += " ".join(value_list)
        else:
            html += "<em style='opacity: 0.6;'>None specified</em>"
        html += "</td></tr>"

    html += "</table>"

    # If reviewed, show comparison
    if use_reviewed:
        html += "<h4 style='margin: 15px 0 10px 0; color: inherit;'>Your Review (shown in edit boxes below):</h4>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"

        for category, values in [
            ("Race", races),
            ("Gender", genders),
            ("Age", ages),
            ("Language", languages),
        ]:
            html += "<tr style='border-bottom: 1px solid rgba(150, 150, 150, 0.3);'>"
            html += f"<td style='padding: 8px; font-weight: bold; width: 120px; color: inherit;'>{category}:</td>"
            html += "<td style='padding: 8px;'>"
            if values:
                value_list = [
                    f"<span style='background-color: rgba(59, 130, 246, 0.2); padding: 2px 8px; border-radius: 4px; margin-right: 5px; border: 1px solid rgba(59, 130, 246, 0.4); color: inherit;'>{val}</span>"
                    for val in values
                ]
                html += " ".join(value_list)
            else:
                html += "<em style='opacity: 0.6;'>None specified</em>"
            html += "</td></tr>"

        html += "</table>"

    html += "</div>"

    # Convert lists to comma-separated strings for text boxes
    race_text = ", ".join(races)
    gender_text = ", ".join(genders)
    age_text = ", ".join(ages)
    lang_text = ", ".join(languages)

    return html, race_text, gender_text, age_text, lang_text


def load_caption_for_item(current_state, item):
    try:
        video_number = item.get("video_number")
        if not video_number:
            return "<em>⚠️ No 'video_number' field found in metadata.</em>"

        topic = current_state.get("topic", "")
        if not topic:
            return "<em>⚠️ No topic loaded.</em>"

        caption_file = os.path.join("captions", topic, f"caption_{video_number}.srt")

        if not os.path.exists(caption_file):
            return f"""<div style="padding: 20px; background-color: rgba(255, 193, 7, 0.15); border-radius: 8px; border: 1px solid rgba(255, 193, 7, 0.3);">
            <strong style="color: #fbbf24;">⚠️ Caption file not found</strong><br>
            Looking for: <code style="background-color: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 3px;">caption_{video_number}.srt</code>
            </div>"""

        with open(caption_file, "r", encoding="utf-8") as f:
            caption_text = f.read()

        if not caption_text.strip():
            return "<em>⚠️ Caption file is empty.</em>"

        return f"""<div style="background-color: rgba(100, 100, 100, 0.2); padding: 20px; border-radius: 8px;
                    max-height: 400px; overflow-y: auto; font-family: 'Courier New', monospace;
                    white-space: pre-wrap; line-height: 1.6; font-size: 14px; border: 1px solid rgba(150, 150, 150, 0.3);">
{caption_text}
</div>"""

    except Exception as e:
        return f"<div style='padding: 20px; background-color: rgba(239, 68, 68, 0.15); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);'><strong style='color: #f87171;'>❌ Error:</strong> {str(e)}</div>"


def move(delta, race_text, gender_text, age_text, lang_text, current_state):
    """Auto-saves current demographics before moving to next/previous item."""
    if not current_state or not current_state.get("items"):
        return (
            "<em>Load a topic first.</em>",
            "<em>No caption.</em>",
            "—",
            "—",
            "",
            "",
            "",
            "",
            "",
            "",
            current_state,
        )

    # Auto-save current item's demographics before moving
    save_msg, current_state = update_demographics(
        race_text, gender_text, age_text, lang_text, current_state, silent=True
    )

    update_session_activity(current_state["session_id"])
    new_idx = current_state["idx"] + delta
    current_state["idx"] = max(0, min(len(current_state["items"]) - 1, new_idx))

    return show_current(current_state) + ("", current_state)  # Empty save message


def update_demographics(
    race_text, gender_text, age_text, lang_text, current_state, silent=False
):
    """Updates demographics for the current item - saves to demographics_detailed_reviewed."""
    if not current_state or not current_state.get("items"):
        return "❌ Load a topic first.", current_state

    update_session_activity(current_state["session_id"])
    idx = current_state["idx"]
    item = current_state["items"][idx]

    # Parse comma-separated values and clean them
    def parse_list(text):
        if not text or not text.strip():
            return []
        return [v.strip() for v in text.split(",") if v.strip()]

    races = parse_list(race_text)
    genders = parse_list(gender_text)
    ages = parse_list(age_text)
    languages = parse_list(lang_text)

    # Check if anything changed from the current reviewed version (or original if not reviewed yet)
    current_reviewed = item.get("demographics_detailed_reviewed", {})
    current_original = item.get("demographics_detailed", {})
    compare_against = current_reviewed if current_reviewed else current_original

    changed = (
        races != compare_against.get("race", [])
        or genders != compare_against.get("gender", [])
        or ages != compare_against.get("age", [])
        or languages != compare_against.get("language", [])
    )

    if not changed and not silent:
        return "ℹ️ No changes detected.", current_state

    # Save to demographics_detailed_reviewed (keeps original AI annotation intact)
    item["demographics_detailed_reviewed"] = {
        "race": races,
        "gender": genders,
        "age": ages,
        "language": languages,
    }

    # Add review metadata
    item["demographics_review_info"] = {
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": current_state.get("session_id", "unknown"),
    }

    write_back(current_state)
    _pending_ops.add((current_state["json_path"], current_state["repo_path"]))

    if silent:
        return "", current_state
    return f"✅ Demographics reviewed and saved for item {idx + 1}", current_state


def write_back(current_state):
    root = current_state["root"]
    items = current_state["items"]
    path = current_state["path"]

    if path == "<root>":
        data_to_write = items
    else:
        root[path] = items
        data_to_write = root

    with open(current_state["json_path"], "w", encoding="utf-8") as f:
        json.dump(data_to_write, f, ensure_ascii=False, indent=2)


def get_session_status(current_state):
    current_state = initialize_state(current_state)
    sessions = get_active_sessions_info()
    if not sessions:
        return "✅ **No other active sessions.** Safe to push."
    status = f"**Active Sessions ({len(sessions)}):**\n"
    for sid, info in sessions.items():
        marker = "👤 **YOU**" if sid == current_state.get("session_id") else "👥 Other"
        age = int(time.time() - info["last_active"])
        status += f"- {marker}: Session `{sid}` on topic **{info['topic']}** (active {age}s ago)\n"
    return status


def push_to_dataset(current_state):
    if not (HF_TOKEN and DATA_REPO_ID):
        return "⚠️ Push failed: HF_TOKEN or DATA_REPO_ID secrets are not set."
    if not _pending_ops:
        return "✅ Nothing to push. All changes are already saved."
    try:
        operations = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=l)
            for (l, p) in sorted(_pending_ops)
        ]
        create_commit(
            repo_id=DATA_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Demographics review update from session {current_state.get('session_id', 'unknown')}",
            token=HF_TOKEN,
        )
        num_files = len(_pending_ops)
        _pending_ops.clear()
        return f"✅ **Success!** Pushed {num_files} file(s) to the dataset."
    except Exception as e:
        return f"❌ **Push Failed:** {e}"


# --- 4. Helper Functions ---
def to_embed_html(url):
    if not isinstance(url, str):
        return "<em>Invalid URL in data.</em>"

    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        try:
            parsed_url = urllib.parse.urlparse(url)
            if "youtu.be" in u:
                video_id = parsed_url.path[1:]
            else:
                video_id = urllib.parse.parse_qs(parsed_url.query)["v"][0]

            if video_id:
                return (
                    f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" '
                    'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
                    'gyroscope; picture-in-picture" allowfullscreen></iframe>'
                )
        except Exception:
            return "<em>Could not parse YouTube URL.</em>"
    return f"<em>Video URL: {url}</em>"


def path_relative_to_repo(p):
    return os.path.relpath(p, os.getcwd())


def scan_topics(base="captions"):
    if not os.path.isdir(base):
        return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])


def choose_json_for_topic(topic, base="captions"):
    """Finds the metadata_enhanced.json file in the topic folder."""
    root = os.path.join(base, topic)

    # Look for metadata_enhanced.json specifically
    enhanced_path = os.path.join(root, "metadata_enhanced.json")
    if os.path.exists(enhanced_path):
        return enhanced_path

    # Fallback to metadata.json if enhanced doesn't exist
    metadata_path = os.path.join(root, "metadata.json")
    if os.path.exists(metadata_path):
        print(
            f"⚠️ Warning: Using metadata.json instead of metadata_enhanced.json for topic '{topic}'"
        )
        return metadata_path

    return None


def best_array_in_json(obj):
    if isinstance(obj, list):
        return "<root>", obj
    if isinstance(obj, dict):
        best_key, best_list = None, []
        for key, value in obj.items():
            if isinstance(value, list) and len(value) > len(best_list):
                best_key, best_list = key, value
        if best_key:
            return best_key, best_list
    return "<root>", []


def detect_id_field(items):
    if not items or not isinstance(items[0], dict):
        return "id"
    keys = items[0].keys()
    for candidate in ["video_id", "videoId", "id", "uid", "uuid"]:
        if candidate in keys:
            return candidate
    return next(iter(keys), "id")


# --- 5. Gradio User Interface ---
with gr.Blocks(title="Demographics Review Tool", theme=gr.themes.Soft()) as demo:
    app_state = gr.State({})

    gr.Markdown("# 📊 Demographics Review Tool")
    gr.Markdown("Review and edit detailed demographic annotations for each video")

    with gr.Column() as login_view:
        passcode_input = gr.Textbox(
            label="Enter Passcode", type="password", placeholder="••••••••"
        )
        login_btn = gr.Button("Unlock")
        login_msg = gr.Markdown("")

    with gr.Column(visible=False) as app_view:
        gr.Markdown(
            "Select a topic, watch the video, review the caption and AI-generated demographics, then edit as needed."
        )
        gr.Markdown(
            "💡 **Tip:** Changes auto-save when you navigate. Original AI annotations are preserved."
        )

        with gr.Row():
            topic_dd = gr.Dropdown(choices=scan_topics(), label="1. Select Topic")
            load_btn = gr.Button("🚀 Load", variant="primary", scale=0)

        status_md = gr.Markdown("*Please load a topic to begin.*")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Video")
                video_preview = gr.HTML("<em>Video will appear here.</em>")

            with gr.Column(scale=1):
                gr.Markdown("### 📄 Caption")
                caption_preview = gr.HTML("<em>Caption will appear here.</em>")

        header_md = gr.Markdown("—")

        # Demographics Display
        demographics_display = gr.HTML("")

        # Demographics Editing Section
        gr.Markdown("---")
        gr.Markdown("### ✏️ Edit Demographics")
        gr.Markdown("""
        **Instructions:** Edit the demographics below. Use comma-separated values for multiple entries.
        **Valid options:**
        - **Race:** White, Black, Asian, Indigenous, Arab, Hispanic
        - **Gender:** Male, Female
        - **Age:** Young (18-24), Middle (25-39), Older adults (40+)
        - **Language:** English, Hindi, Arabic, Spanish, Chinese
        """)

        with gr.Row():
            race_textbox = gr.Textbox(
                label="Race/Ethnicity (comma-separated)",
                placeholder="e.g., White, Asian",
                value="",
            )
            gender_textbox = gr.Textbox(
                label="Gender (comma-separated)",
                placeholder="e.g., Male, Female",
                value="",
            )

        with gr.Row():
            age_textbox = gr.Textbox(
                label="Age Group (comma-separated)",
                placeholder="e.g., Young (18-24), Middle (25-39)",
                value="",
            )
            lang_textbox = gr.Textbox(
                label="Language (comma-separated)",
                placeholder="e.g., English, Spanish",
                value="",
            )

        save_msg = gr.Markdown("")

        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", variant="secondary")
            save_demo_btn = gr.Button("💾 Save (Manual)", variant="primary")
            next_btn = gr.Button("Next ➡️", variant="secondary")

        with gr.Accordion("📄 Topic Information", open=False):
            path_info_md = gr.Markdown("—")

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

    login_btn.click(
        unlock_app, inputs=[passcode_input], outputs=[login_view, app_view, login_msg]
    )

    load_btn.click(
        load_topic,
        [topic_dd, app_state],
        [
            topic_dd,
            status_md,
            video_preview,
            caption_preview,
            header_md,
            path_info_md,
            demographics_display,
            race_textbox,
            gender_textbox,
            age_textbox,
            lang_textbox,
            save_msg,
            app_state,
        ],
    )

    # Previous button with auto-save
    prev_btn.click(
        lambda r, g, a, l, s: move(-1, r, g, a, l, s),
        [race_textbox, gender_textbox, age_textbox, lang_textbox, app_state],
        [
            video_preview,
            caption_preview,
            header_md,
            path_info_md,
            demographics_display,
            race_textbox,
            gender_textbox,
            age_textbox,
            lang_textbox,
            save_msg,
            app_state,
        ],
    )

    # Next button with auto-save
    next_btn.click(
        lambda r, g, a, l, s: move(+1, r, g, a, l, s),
        [race_textbox, gender_textbox, age_textbox, lang_textbox, app_state],
        [
            video_preview,
            caption_preview,
            header_md,
            path_info_md,
            demographics_display,
            race_textbox,
            gender_textbox,
            age_textbox,
            lang_textbox,
            save_msg,
            app_state,
        ],
    )

    # Manual save button
    save_demo_btn.click(
        update_demographics,
        [race_textbox, gender_textbox, age_textbox, lang_textbox, app_state],
        [save_msg, app_state],
    )

    check_btn.click(get_session_status, [app_state], [session_status_md])
    push_btn.click(push_to_dataset, [app_state], [status_md])

if __name__ == "__main__":
    demo.launch()
