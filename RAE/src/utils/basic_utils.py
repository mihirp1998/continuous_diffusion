import cv2
import numpy as np
import ipdb
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import warnings

st = ipdb.set_trace


class TextDataset(Dataset):
    """
    Custom image dataset class that loads images from a directory structure.
    
    Supports both ImageFolder-style structure (with class subdirectories) and
    flat directory structure (all images in one directory).
    
    Args:
        root (str): Root directory path containing images
        transform (callable, optional): Optional transform to be applied on a sample
        class_subdirs (bool): If True, expects ImageFolder structure with class subdirectories.
                             If False, treats root as a flat directory with all images.
    """
    
    def __init__(self, root, transform=None, class_subdirs=True,num_stories=500000, eval_mode=False):
        self.root = root
        self.transform = transform
        # st()
        
        # Load TinyStories dataset
        print("Loading TinyStories dataset...")
        if eval_mode:
            dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        else:
            dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        # st()
        # Convert dataset to list to get length and indexing
        self.stories = []
        # Store stories to disk for faster loading in future runs
        import pickle
        stories_cache_path = os.path.join(self.root, f'stories_cache_{num_stories}_eval_{eval_mode}.pkl')
        # st()
        
        # Try to load from cache first
        if os.path.exists(stories_cache_path):
            print(f"Loading stories from cache: {stories_cache_path}")
            with open(stories_cache_path, 'rb') as f:
                self.stories = pickle.load(f)
            print(f"Loaded {len(self.stories)} stories from cache")
        else:
            print("Converting dataset to list...")
            for i, story in enumerate(dataset):
                if i >= num_stories:  # Limit to first 10k stories for memory
                    break            
                self.stories.append(story['text'])
            print(f"story 0: {self.stories[0]}")
            print(f"story -1: {self.stories[-1]}")
            # create_text_image(self.stories[0])
            
            # Save to cache for next time
            print(f"Saving stories to cache: {stories_cache_path}")
            with open(stories_cache_path, 'wb') as f:
                pickle.dump(self.stories, f)
            print(f"Saved {len(self.stories)} stories to cache")
        # st()
        if len(self.stories) < 8:
            self.stories = self.stories * 1024
        # st()
        print(f"Loaded {len(self.stories)} stories")

    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, idx):
        """
        Get an image created from text and its label.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a PIL Image or transformed tensor,
                   and label is an integer class index
        """
        # print(f"idx: {idx}")
        text = self.stories[idx]
        
        
        try:
            # Create image from text using create_text_image            
            img_array, txt_array = create_text_image(text)
            
            if img_array is None:
                # If text is too long, try next story
                if idx < len(self.stories) - 1:
                    return self.__getitem__(idx + 1)
                else:
                    return self.__getitem__(0)
            
            # Convert BGR to RGB (cv2 uses BGR, PIL uses RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            from PIL import Image
            image = Image.fromarray(img_array)
            
        except Exception as e:
            # If image creation fails, try to load a different story
            warnings.warn(f"Failed to create image for story {idx}: {e}. Trying next story.")
            if idx < len(self.stories) - 1:
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, text


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
    # st()
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
                    # print(f"Reached max lines: {len(lines)}")
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
                            # print(f"Reached max lines: {len(lines)}")    
                            break_because_of_max_lines = True
                            break
                        lines.append(word[j:j+chars_per_line])
                    continue
            current_line = [word]
    # st()
    # Add remaining line if space
    if current_line and len(lines) < max_lines:
        lines.append(' '.join(current_line))
    
    if break_because_of_max_lines:
        return None, text
    
    # Draw text line by line
    for i, line in enumerate(lines):
        y_position = y_start + i * line_height
        cv2.putText(img, line, (margin, y_position), 
                   font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return img, text
