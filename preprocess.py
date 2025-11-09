import av
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel
import os
import json

# --- Configuration ---
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
# VideoMAE processes 16 frames at a time
NUM_FRAMES_IN_CLIP = 16

# Define your label-to-integer mapping
# Make sure this matches your ASFormer model's "num_classes"
LABEL_MAP = {
    "nonseizure": 0,
    "preseizure": 1,
    "seizure": 2
    # Add any other classes you defined
}
DEFAULT_LABEL = 0 # This is 'nonseizure'

# --- Load Model (do this once) ---
print("Loading VideoMAE model...")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEModel.from_pretrained(MODEL_ID)
model.eval()  # Set to evaluation mode
print("Model loaded.")

def extract_features_from_video(video_path):
    """
    Extracts features from a video file, clip by clip.
    
    A "clip" is 16 frames. We process the video, 16 frames at a time,
    and get one feature vector for each 16-frame clip.
    """
    print(f"Processing video: {video_path}")
    
    # Use PyAV to open the video
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"Error opening video {video_path}: {e}")
        return None, 0.0

    clip_features = []
    frames = []
    
    video_stream = container.streams.video[0]
    fps = float(video_stream.average_rate)

    try:
        # Loop through all frames in the video
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            
            # When we have enough frames for a clip, process them
            if len(frames) == NUM_FRAMES_IN_CLIP:
                # 1. Prepare the clip for the model
                inputs = processor(list(frames), return_tensors="pt")
                
                # 2. Run the clip through VideoMAE
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # 3. Get the feature vector
                features = outputs.last_hidden_state.mean(dim=1)
                clip_features.append(features)
                
                # Clear the frames list for the next clip
                frames = []
                
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
    finally:
        container.close()

    if not clip_features:
        print(f"Warning: No features extracted from {video_path}. Video might be too short or unreadable.")
        return None, fps

    # Stack all clip-features into a single tensor
    final_feature_tensor = torch.cat(clip_features, dim=0)
    print(f"Done. Extracted feature tensor of shape: {final_feature_tensor.shape}")
    
    return final_feature_tensor, fps

def create_label_tensor(label_json_path, num_clips, fps):
    """
    Generates a per-clip label tensor from a Label Studio JSON export.
    """
    
    # Calculate how many seconds each feature-clip represents
    if fps == 0:
        print("Error: Cannot create labels with 0 FPS.")
        return None
        
    seconds_per_clip = NUM_FRAMES_IN_CLIP / fps

    # Load the exported annotations
    try:
        with open(label_json_path, 'r') as f:
            # The structure might vary slightly based on your Label Studio setup
            # This assumes the root is a list of segments
            annotations = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {label_json_path}: {e}")
        return None

    # Initialize all clips with the default label (0 = nonseizure)
    label_array = np.full(num_clips, DEFAULT_LABEL, dtype=np.int64)

    # The Core Mapping Logic
    for clip_index in range(num_clips):
        # Find the middle-point time of this clip
        clip_mid_time_sec = (clip_index * seconds_per_clip) + (seconds_per_clip / 2)
        
        # Check if this time falls into any annotated segment
        for segment in annotations:
            # Adjust keys based on your JSON export (e.g., 'value' 'start', 'end')
            try:
                start_time = segment['value']['start']
                end_time = segment['value']['end']
                # Assumes one label per segment
                label_name = segment['value']['labels'][0] 
            except KeyError:
                print(f"Warning: Skipping malformed segment in {label_json_path}: {segment}")
                continue
            
            if clip_mid_time_sec >= start_time and clip_mid_time_sec <= end_time:
                if label_name in LABEL_MAP:
                    label_array[clip_index] = LABEL_MAP[label_name]
                else:
                    print(f"Warning: Unknown label '{label_name}' in {label_json_path}. Using default.")
                
                break # Move to the next clip once a match is found

    return torch.from_numpy(label_array)


# --- Main execution ---
if __name__ == "__main__":

    VIDEO_INPUT_DIR = "./videos"
    ANNOTATION_INPUT_DIR = "./video_annotations"
    FEATURE_OUTPUT_DIR = "./preprocessed_videos"
    LABEL_OUTPUT_DIR = "./final_labels"
    
    # Ensure the output directories exist
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)
    
    # List all files in the input folder
    for filename in os.listdir(VIDEO_INPUT_DIR):
        # Check for common video file extensions
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            
            base_filename = os.path.splitext(filename)[0]
            video_file_path = os.path.join(VIDEO_INPUT_DIR, filename)
            
            # --- Check for corresponding annotation file ---
            json_filename = f"{base_filename}.json"
            label_file_path = os.path.join(ANNOTATION_INPUT_DIR, json_filename)
            
            if not os.path.exists(label_file_path):
                print(f"Warning: No label file found at {label_file_path}. Skipping video {filename}.")
                continue
            
            print(f"--- Processing {filename} ---")
            
            # 1. Process your video to get features
            feature_tensor, video_fps = extract_features_from_video(video_file_path)
            
            # 2. Save the features
            if feature_tensor is not None:
                # Save the feature tensor
                feature_save_path = os.path.join(FEATURE_OUTPUT_DIR, f"{base_filename}.pt")
                torch.save(feature_tensor, feature_save_path)
                print(f"Features saved to {feature_save_path}")
                
                # 3. Create and save the corresponding label tensor
                num_clips = feature_tensor.shape[0]
                label_tensor = create_label_tensor(label_file_path, num_clips, video_fps)
                
                if label_tensor is not None:
                    label_save_path = os.path.join(LABEL_OUTPUT_DIR, f"{base_filename}.pt")
                    torch.save(label_tensor, label_save_path)
                    print(f"Labels saved to {label_save_path}")
                else:
                    print(f"Error: Failed to create labels for {filename}.")
            
            else:
                print(f"Skipped saving features for {filename} (extraction failed or video empty)")
        else:
            print(f"Skipping non-video file: {filename}")