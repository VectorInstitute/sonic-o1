import whisperx
from pathlib import Path

# Paths
audio_path = "dataset/audios/01_Patient-Doctor_Consultations/audio_020.m4a"
original_srt = "dataset/captions/01_Patient-Doctor_Consultations/caption_020.srt"
output_srt = "test_caption_020.srt"

print("Loading model (large-v3 for best quality)...")
model = whisperx.load_model('large-v3', device='cuda', compute_type='float16')

print("Loading audio...")
audio = whisperx.load_audio(audio_path)

print("Transcribing...")
# Force English language
result = model.transcribe(audio, batch_size=16, language='en')

print(f"Language: en")
print(f"Segments: {len(result['segments'])}")

# Align for better timestamps
print("Aligning timestamps...")
model_a, metadata = whisperx.load_align_model(language_code='en', device="cuda")
result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda")

# Format as SRT
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

srt_lines = []
for i, segment in enumerate(result["segments"], 1):
    start = format_timestamp(segment['start'])
    end = format_timestamp(segment['end'])
    text = segment['text'].strip()
    
    srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

# Save
with open(output_srt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(srt_lines))

print(f"\n✅ Generated caption saved to: {output_srt}")
print("\n--- First 3 segments ---")
for segment in result["segments"][:3]:
    print(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}")
    print(f"{segment['text']}\n")

print(f"\n--- Original caption (first 3) ---")
with open(original_srt, 'r') as f:
    print(f.read()[:500])