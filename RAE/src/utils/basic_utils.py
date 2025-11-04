import cv2
import numpy as np
import ipdb
st = ipdb.set_trace


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
