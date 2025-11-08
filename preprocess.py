import av
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel

# --- Configuration ---
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
# VideoMAE processes 16 frames at a time
NUM_FRAMES_IN_CLIP = 16

# --- Load Model (do this once) ---
print("Loading models...")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEModel.from_pretrained(MODEL_ID)
model.eval()  # Set to evaluation mode
print("Models loaded.")

def extract_features_from_video(video_path):
    """
    Extracts features from a video file, clip by clip.
    
    A "clip" is 16 frames. We process the video, 16 frames at a time,
    and get one feature vector for each 16-frame clip.
    """
    print(f"Processing video: {video_path}")
    
    # Use PyAV to open the video
    container = av.open(video_path)
    
    clip_features = []
    frames = []

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
                # We get the 'last_hidden_state' and average it to get
                # a single representative vector for this 16-frame clip.
                # (batch_size, num_patches, feature_dim) -> (batch_size, feature_dim)
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
        return None

    # Stack all clip-features into a single tensor
    # Shape will be [num_clips, feature_dimension]
    # e.g., for a 2-minute video: [450, 768]
    final_feature_tensor = torch.cat(clip_features, dim=0)
    print(f"Done. Extracted feature tensor of shape: {final_feature_tensor.shape}")
    
    return final_feature_tensor

# --- How to use it ---
if __name__ == "__main__":
    # 1. Process your video
    # Replace with the path to one of your labeled videos
    video_file_path = "path/to/your/seizure_video_01.mp4" 
    feature_tensor = extract_features_from_video(video_file_path)
    
    # 2. Save the features
    if feature_tensor is not None:
        # You will feed this file to ASFormer
        save_path = "video_01_features.pt" 
        torch.save(feature_tensor, save_path)
        print(f"Features saved to {save_path}")
