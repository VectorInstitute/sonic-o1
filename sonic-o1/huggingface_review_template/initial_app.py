import glob
import json
import os
import time
import urllib.parse
import uuid

import gradio as gr
from huggingface_hub import CommitOperationAdd, create_commit, snapshot_download


# --- 1. Configuration and Initial Data Sync ---
# Load secrets from the Space's settings
PASSCODE = os.environ.get("PASSCODE")
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO_ID = os.environ.get("DATA_REPO_ID")  # e.g., "your-username/your-dataset-repo"


# On startup, download the 'videos' folder from your dataset repo to the Space's local storage.
if DATA_REPO_ID and snapshot_download:
    print(f"Syncing data from dataset repo: {DATA_REPO_ID}")
    try:
        snapshot_download(
            repo_id=DATA_REPO_ID,
            repo_type="dataset",
            local_dir=".",  # Download to the root, which will create the 'videos' folder
            token=HF_TOKEN,
            allow_patterns="videos/**",  # Only download the 'videos' folder and its contents
        )
        print("Data sync complete.")
    except Exception as e:
        print(f"Could not sync data from {DATA_REPO_ID}: {e}")
else:
    print("Skipping data sync: DATA_REPO_ID secret is not set.")

# --- 2. Session Management (for tracking active files) ---
SESSIONS_DIR = "sessions"
SESSION_TIMEOUT = 300  # 5 minutes in seconds
os.makedirs(SESSIONS_DIR, exist_ok=True)
_pending_ops = set()  # This remains global as it's a queue for the single push process


def get_session_file(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def register_session(session_id, repo_path, topic):
    try:
        session_data = {"file": repo_path, "last_active": time.time(), "topic": topic}
        with open(get_session_file(session_id), "w") as f:
            json.dump(session_data, f, indent=2)
        print(f"Registered session {session_id} for topic '{topic}'")
    except Exception as e:
        print(f"Error registering session {session_id}: {e}")


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
                os.remove(session_file)  # Clean up stale session file
        except Exception:
            pass  # Ignore corrupted session files
    return active_sessions


# --- 3. Core Application Logic (using gr.State) ---

QUALITY_KEY = "Qualitylabel"


def initialize_state(current_state):
    """Initializes a user's state if it's empty."""
    if not isinstance(current_state, dict) or not current_state:
        return {"session_id": str(uuid.uuid4())[:8], "items": [], "idx": 0}
    return current_state


def load_topic(topic, current_state):
    """Loads a topic's JSON data into the user's state."""
    current_state = initialize_state(current_state)
    print(f"Loading topic '{topic}' for session {current_state['session_id']}")

    jp = choose_json_for_topic(topic)
    if not jp:
        msg = f"No JSON file found for topic '{topic}'."
        return gr.update(), msg, "<em>No items.</em>", "—", "—", "—", current_state

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
        html, header, topic_file, meta = show_current(current_state)
        return (
            gr.update(choices=scan_topics(), value=topic),
            status,
            html,
            header,
            topic_file,
            meta,
            current_state,
        )
    except Exception as e:
        return (
            gr.update(),
            f"❌ Error loading JSON: {e}",
            "<em>Error</em>",
            "—",
            "—",
            "—",
            current_state,
        )


def show_current(current_state):
    """Generates the UI components for the current item in the user's state."""
    if not current_state or not current_state.get("items"):
        return "<em>Load a topic to begin.</em>", "—", "—", "—"

    idx = current_state["idx"]
    items = current_state["items"]
    item = items[idx]

    id_field = detect_id_field(items)
    url_field = detect_url_field(items)

    item_id = str(item.get(id_field, idx))
    url = item.get(url_field)
    html = to_embed_html(url)
    label = item.get(QUALITY_KEY, "(unlabeled)")
    meta = json.dumps(item, ensure_ascii=False, indent=2)
    header = f"Item {idx + 1} / {len(items)} | ID: {item_id} | Status: {label}"
    topic_file = f"Topic: {current_state.get('topic', 'N/A')}"
    return html, header, topic_file, meta


def move(delta, current_state):
    """Moves to the previous/next item."""
    if not current_state or not current_state.get("items"):
        return "<em>Load a topic first.</em>", "—", "—", "—", current_state

    update_session_activity(current_state["session_id"])
    new_idx = current_state["idx"] + delta
    current_state["idx"] = max(0, min(len(current_state["items"]) - 1, new_idx))

    html, header, topic_file, meta = show_current(current_state)
    return html, header, topic_file, meta, current_state


def set_label(value, current_state):
    """Sets a label for the current item and saves it."""
    if not current_state or not current_state.get("items"):
        return "<em>Load a topic first.</em>", "—", "—", "—", current_state

    update_session_activity(current_state["session_id"])

    idx = current_state["idx"]
    current_state["items"][idx][QUALITY_KEY] = value

    write_back(current_state)
    _pending_ops.add((current_state["json_path"], current_state["repo_path"]))

    return move(+1, current_state)


def write_back(current_state):
    """Writes the modified data back to the local JSON file."""
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
    """Checks for other active users."""
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
    """Pushes all queued local changes to the dataset repository."""
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
            commit_message=f"Data update from session {current_state.get('session_id', 'unknown')}",
            token=HF_TOKEN,
        )
        num_files = len(_pending_ops)
        _pending_ops.clear()
        return f"✅ **Success!** Pushed {num_files} file(s) to the dataset. No restart needed."
    except Exception as e:
        return f"❌ **Push Failed:** {e}"


# --- 4. Helper Functions ---


def to_embed_html(url):
    """
    FIXED: This is the restored, fully-functional video embedder.
    It correctly handles both direct video links and YouTube URLs.
    """
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

    return f'<video src="{url}" controls style="width: 100%; max-height: 400px; border-radius: 8px;"></video>'


def path_relative_to_repo(p):
    return os.path.relpath(p, os.getcwd())


def scan_topics(base="videos"):
    if not os.path.isdir(base):
        return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])


def choose_json_for_topic(topic, base="videos"):
    root, best_path, best_len = os.path.join(base, topic), None, -1
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".json"):
                full_path = os.path.join(dirpath, fn)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    _, items = best_array_in_json(data)
                    if len(items) > best_len:
                        best_len, best_path = len(items), full_path
                except (IOError, json.JSONDecodeError):
                    continue
    return best_path


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


def detect_url_field(items):
    if not items or not isinstance(items[0], dict):
        return "url"
    keys = items[0].keys()
    for candidate in ["video_url", "videoUrl", "url", "src", "path", "file"]:
        if candidate in keys and is_probably_url(items[0].get(candidate)):
            return candidate
    for key in keys:
        if is_probably_url(items[0].get(key)):
            return key
    return "url"


def is_probably_url(v):
    return isinstance(v, str) and (
        v.lower().startswith("http") or v.lower().endswith((".mp4", ".webm", ".mov"))
    )


# --- 5. Gradio User Interface ---
with gr.Blocks(title="Video Review Tool", theme=gr.themes.Soft()) as demo:
    app_state = gr.State({})

    gr.Markdown("# 🎬 Video Quality Review Tool")

    with gr.Column() as login_view:
        passcode_input = gr.Textbox(
            label="Enter Passcode", type="password", placeholder="••••••••"
        )
        login_btn = gr.Button("Unlock")
        login_msg = gr.Markdown("")

    with gr.Column(visible=False) as app_view:
        gr.Markdown(
            "Select a topic, label videos as **Good** or **Bad**, and push your changes when done."
        )
        with gr.Row():
            topic_dd = gr.Dropdown(choices=scan_topics(), label="1. Select Topic")
            load_btn = gr.Button("🚀 Load", variant="primary", scale=0)
        status_md = gr.Markdown("*Please load a topic to begin.*")

        with gr.Row():
            with gr.Column(scale=2):
                preview_html = gr.HTML("<em>Video will appear here.</em>")
                header_md = gr.Markdown("—")
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous")
                    next_btn = gr.Button("Next ➡️")
                with gr.Row():
                    good_btn = gr.Button("👍 Good", variant="primary")
                    bad_btn = gr.Button("👎 Bad", variant="secondary")
            with gr.Column(scale=1):
                path_info_md = gr.Markdown("—")
                meta_code = gr.Code(
                    label="Item JSON Data", language="json", interactive=False
                )

        gr.Markdown("---")
        gr.Markdown("### 3. Push Changes to Dataset")
        gr.Markdown(
            "Check for other active users, then push your saved changes. **This will not restart the app.**"
        )
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
            preview_html,
            header_md,
            path_info_md,
            meta_code,
            app_state,
        ],
    )
    prev_btn.click(
        lambda s: move(-1, s),
        [app_state],
        [preview_html, header_md, path_info_md, meta_code, app_state],
    )
    next_btn.click(
        lambda s: move(+1, s),
        [app_state],
        [preview_html, header_md, path_info_md, meta_code, app_state],
    )
    good_btn.click(
        lambda s: set_label("Good", s),
        [app_state],
        [preview_html, header_md, path_info_md, meta_code, app_state],
    )
    bad_btn.click(
        lambda s: set_label("Bad", s),
        [app_state],
        [preview_html, header_md, path_info_md, meta_code, app_state],
    )
    check_btn.click(get_session_status, [app_state], [session_status_md])
    push_btn.click(push_to_dataset, [app_state], [status_md])

if __name__ == "__main__":
    demo.launch()
