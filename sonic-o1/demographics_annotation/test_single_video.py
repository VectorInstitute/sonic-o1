"""
Test script to process a single media file with multimodal support (video + audio + captions)
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config_loader import Config
from model import DemographicsAnnotator


def test_single_media():
    """Test processing of a single item with multimodal support (video, audio, captions)"""
    
    # Load configuration
    config_path = "config.yaml"
    config = Config(config_path)
    
    # Set up test parameters
    topic = "01_Patient-Doctor_Consultations"
    video_number = "015"  # Item identifier
    
    print("=" * 80)
    print("MULTIMODAL DEMOGRAPHICS ANNOTATION TEST")
    print("=" * 80)
    print(f"Topic: {topic}")
    print(f"Item Number: {video_number}")
    print(f"Model: {config.model_name}")
    print("=" * 80)
    
    # Check API key
    if not config.api_key:
        print("ERROR: API key not set!")
        print("Please set GEMINI_API_KEY environment variable or update config.yaml")
        return
    
    # Get file paths
    paths = config.get_topic_paths(topic)
    video_path = config.get_file_path(topic, "video", video_number)
    audio_path = config.get_file_path(topic, "audio", video_number)
    caption_path = config.get_file_path(topic, "caption", video_number)
    metadata_path = paths["metadata"]
    
    print("\nFile Paths:")
    print(f"  Video:    {video_path}")
    print(f"  Audio:    {audio_path}")
    print(f"  Caption:  {caption_path}")
    print(f"  Metadata: {metadata_path}")
    
    # Check which files exist
    print("\nChecking available modalities...")
    has_video = video_path.exists()
    has_audio = audio_path.exists()
    has_caption = caption_path.exists()
    has_metadata = metadata_path.exists()
    
    modalities = []
    
    if has_video:
        video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"  [+] Video: {video_path.name} ({video_size:.2f} MB)")
        modalities.append("video")
    else:
        print(f"  [-] Video: Not found")
    
    if has_audio:
        audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        print(f"  [+] Audio: {audio_path.name} ({audio_size:.2f} MB)")
        modalities.append("audio")
    else:
        print(f"  [-] Audio: Not found")
    
    if has_caption:
        caption_size = os.path.getsize(caption_path) / 1024  # KB
        print(f"  [+] Caption: {caption_path.name} ({caption_size:.2f} KB)")
        modalities.append("captions")
    else:
        print(f"  [-] Caption: Not found")
    
    if has_metadata:
        print(f"  [+] Metadata: Found")
    else:
        print(f"  [-] Metadata: Not found")
        return
    
    # Check if we have at least video or audio
    if not has_video and not has_audio:
        print("\nERROR: No media files found!")
        print("  Need at least video OR audio to proceed.")
        return
    
    # Summary of what will be processed
    print(f"\n{'='*80}")
    print(f"PROCESSING MODE: Multimodal ({' + '.join(modalities)})")
    print(f"{'='*80}")
    
    # Apply preference logic
    if has_video and has_audio and config.prefer_video_with_audio:
        print(f"INFO: Config preference set to use video's embedded audio only")
        print(f"      Separate audio file will be ignored")
        has_audio = False  # Don't send separate audio
    elif has_video and has_audio:
        print(f"INFO: Will send both video AND separate audio to Gemini")
    
    # Load metadata for this item
    print("\nLoading metadata...")
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    # Find item metadata
    media_metadata = None
    for item in metadata_list:
        if item.get('video_number') == video_number:
            media_metadata = item
            break
    
    if not media_metadata:
        print(f"WARNING: No metadata found for item_{video_number}, using first entry")
        media_metadata = metadata_list[0] if metadata_list else {}
    
    print(f"  Title: {media_metadata.get('title', 'Unknown')}")
    print(f"  Duration: {media_metadata.get('duration_seconds', 0)} seconds")
    print(f"  Topic: {media_metadata.get('topic_name', 'Unknown')}")
    
    # Load caption preview if available
    caption_text = ""
    if has_caption:
        print("\nCaption Preview:")
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption_text = f.read()
            preview = caption_text[:300] + "..." if len(caption_text) > 300 else caption_text
            print("-" * 80)
            print(preview)
            print("-" * 80)
            print(f"Total caption length: {len(caption_text)} characters")
    
    # Initialize annotator
    print("\n" + "=" * 80)
    print("STARTING MULTIMODAL ANNOTATION")
    print("=" * 80)
    
    try:
        annotator = DemographicsAnnotator(config)
        print("SUCCESS: Annotator initialized")
        
        # Process media with multimodal support
        modality_count = sum([has_video, has_audio, has_caption])
        print(f"\nProcessing with {modality_count} modalities...")
        if has_video:
            print(f"  > Video analysis enabled")
        if has_audio:
            print(f"  > Separate audio analysis enabled")
        if has_caption:
            print(f"  > Transcript/caption analysis enabled")
        
        print(f"\nProcessing... (this may take 30-120 seconds)")
        
        start_time = datetime.now()
        
        demographics = annotator.process_media(
            video_path=video_path if has_video else None,
            audio_path=audio_path if has_audio else None,
            transcript_path=caption_path if has_caption else None,
            metadata=media_metadata,
            config=config
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\nSUCCESS: Processing completed in {processing_time:.2f} seconds")
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if "error" in demographics.get("demographics_annotation", {}):
            print("ERROR in processing:")
            print(demographics["demographics_annotation"].get("explanation", "Unknown error"))
        else:
            print("\nDETAILED DEMOGRAPHICS:")
            print("-" * 80)
            detailed = demographics["demographics_detailed"]
            for category, values in detailed.items():
                print(f"{category.upper():12} : {values if values else '[]'}")
            
            print("\nCONFIDENCE SCORES:")
            print("-" * 80)
            confidence = demographics["demographics_confidence"]
            for category, scores in confidence.items():
                if scores:
                    print(f"{category.upper():12} :")
                    for item, score in scores.items():
                        print(f"  - {item:20} : {score:.2f}")
                else:
                    print(f"{category.upper():12} : {{}}")
            
            print("\nANNOTATION METADATA:")
            print("-" * 80)
            annotation = demographics["demographics_annotation"]
            print(f"Individuals Count  : {annotation.get('individuals_count', 0)}")
            print(f"Modalities Used    : {annotation.get('modalities_used', 'N/A')}")
            print(f"Model              : {annotation.get('model', 'Unknown')}")
            print(f"Timestamp          : {annotation.get('annotated_at', 'Unknown')}")
            print(f"\nExplanation:")
            print(f"  {annotation.get('explanation', 'No explanation')}")
        
        # Save test output
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"test_{topic}_{video_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        test_result = {
            "test_info": {
                "test_date": datetime.now().isoformat(),
                "topic": topic,
                "item_number": video_number,
                "modalities_available": {
                    "video": has_video,
                    "audio": has_audio,
                    "caption": has_caption
                },
                "files_processed": {
                    "video": str(video_path) if has_video else None,
                    "audio": str(audio_path) if has_audio else None,
                    "caption": str(caption_path) if has_caption else None
                },
                "processing_time_seconds": processing_time
            },
            "media_metadata": media_metadata,
            "demographics_result": demographics
        }
        
        with open(output_file, 'w') as f:
            json.dump(test_result, f, indent=2)
        
        print(f"\nSUCCESS: Test results saved to: {output_file}")
        
        # Validate response structure
        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)
        
        required_fields = ["demographics_detailed", "demographics_confidence", "demographics_annotation"]
        for field in required_fields:
            if field in demographics:
                print(f"[+] {field} present")
            else:
                print(f"[-] {field} MISSING")
        
        # Check for reasonable results
        detailed = demographics.get("demographics_detailed", {})
        if detailed:
            total_demographics = sum(len(v) for v in detailed.values())
            if total_demographics > 0:
                print(f"\n[+] Found {total_demographics} demographic attributes")
            else:
                print("\n[!] No demographic attributes found - check media content")
        
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_single_media()