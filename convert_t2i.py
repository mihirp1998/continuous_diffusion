from datasets import load_dataset
import cv2
import numpy as np
import textwrap
from tqdm import tqdm
import ipdb
st = ipdb.set_trace
import os

root_dir = "/data/user_data/mprabhud/tiny_story_dataset"
root_dir  = "/home/mprabhud/datasets/tinystories"
# Load TinyStories dataset
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Create output directory
# os.makedirs("tinystories_images", exist_ok=True)

os.makedirs(f"{root_dir}/image_dataset", exist_ok=True)
os.makedirs(f"{root_dir}/text_dataset", exist_ok=True)

# Function to create image from text
def create_text_image(text, image_size=512, font_scale=0.6, font_thickness=1):
    # Create white background
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    # Calculate text positioning
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    available_width = image_size - 2 * margin
    
    # Calculate line height and start position
    (text_width, text_height), baseline = cv2.getTextSize("Ag", font, font_scale, font_thickness)
    line_height = text_height + baseline + 5
    y_start = 30
    
    # Calculate maximum number of lines that can fit
    max_lines = (image_size - y_start - 30) // line_height
    
    # Wrap text based on actual pixel width
    words = text.split()
    lines = []
    current_line = []
    break_because_of_max_lines = False
    
    for word in words:
        # Test if adding this word fits
        test_line = ' '.join(current_line + [word])
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        
        if line_width <= available_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                if len(lines) >= max_lines:
                    print(f"Reached max lines: {len(lines)}")
                    break_because_of_max_lines = True
                    break
            # Handle very long words that don't fit even alone
            if current_line == []:  # Word itself is too long
                (word_width, _), _ = cv2.getTextSize(word, font, font_scale, font_thickness)
                if word_width > available_width:
                    # Split long word (approximate)
                    chars_per_line = int(len(word) * available_width / word_width) or 1
                    for j in range(0, len(word), chars_per_line):
                        if len(lines) >= max_lines:
                            print(f"Reached max lines: {len(lines)}")    
                            break_because_of_max_lines = True
                            break
                        lines.append(word[j:j+chars_per_line])
                    continue
            current_line = [word]
    
    # Add remaining line if space
    if current_line and len(lines) < max_lines:
        lines.append(' '.join(current_line))
    
    if break_because_of_max_lines:
        return None
    
    # Draw text line by line
    for i, line in enumerate(lines):
        y_position = y_start + i * line_height
        cv2.putText(img, line, (margin, y_position), 
                   font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return img
# st()

# Generate images from first 10 stories
print("Creating images from TinyStories...")
for i, story in enumerate(tqdm(dataset)):
    if i >= 10:  # Limit to first 10 stories
        break
    
    text = story['text']
    img = create_text_image(text)
    
    if img is not None:
        # Save image
        filename = f"{root_dir}/image_dataset/story_{i+1:08d}.png"
        cv2.imwrite(filename, img)
        with open(f"{root_dir}/text_dataset/story_{i+1:08d}.txt", "w") as f:
            f.write(text)
        # print(f"Saved: {filename}")

# print("Done! Images saved in 'tinystories_images' directory")
