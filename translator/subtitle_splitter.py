import os
import re
from datetime import datetime, timedelta

# Removes bracketed annotations (like [Music], [Applause]) and trims whitespace
def clean_text_lines(text_lines):
    cleaned_lines = []
    for line in text_lines:
        line_clean = re.sub(r'\[.*?\]', '', line).strip()
        cleaned_lines.append(line_clean)
    return cleaned_lines

# Parses SRT timestamp string into a datetime object
def parse_time(s):
    return datetime.strptime(s, "%H:%M:%S,%f")

# Formats a datetime object back into SRT timestamp format
def format_time(t):
    return t.strftime("%H:%M:%S,%f")[:-3]  # Remove last 3 digits to match SRT precision

# Splits a time interval into two equal halves
def split_time(start, end):
    midpoint = start + (end - start) / 2
    return (start, midpoint), (midpoint, end)

# Splits a list of text lines into two halves
def split_text_lines(lines):
    mid = len(lines) // 2
    return lines[:mid], lines[mid:]

# Processes and enhances an SRT file by cleaning, splitting, and merging blocks
def split_srt_blocks(input_path, output_path):
    # Read the entire subtitle file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into blocks separated by blank lines
    blocks = re.split(r'\n\s*\n', content.strip())
    temp_blocks = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue  # Skip incomplete blocks

        time_line = lines[1]
        text_lines = clean_text_lines(lines[2:])  # Remove tags and clean text

        # Parse the start and end times of the subtitle block
        times = time_line.split(' --> ')
        try:
            start = parse_time(times[0].strip())
            end = parse_time(times[1].strip())
        except Exception as e:
            print(f"Skipping invalid time format in {input_path}: {time_line}")
            continue

        # Split block if it contains multiple lines of text
        if len(text_lines) > 1:
            (start1, end1), (start2, end2) = split_time(start, end)
            part1, part2 = split_text_lines(text_lines)

            temp_blocks.append((start1, end1, part1))
            temp_blocks.append((start2, end2, part2))
        else:
            temp_blocks.append((start, end, text_lines))

    # Merge consecutive blocks with identical text lines
    merged_blocks = []
    for start, end, lines in temp_blocks:
        if not merged_blocks:
            merged_blocks.append((start, end, lines))
        else:
            prev_start, prev_end, prev_lines = merged_blocks[-1]
            if lines == prev_lines:
                # Extend previous block's end time if text is identical
                merged_blocks[-1] = (prev_start, end, prev_lines)
            else:
                merged_blocks.append((start, end, lines))

    # Write enhanced subtitle file with updated indexing and formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (start, end, lines) in enumerate(merged_blocks, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n")

# Ensures a filename is safe for use on any OS by removing illegal characters
def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

# Main function to process an input subtitle file and enhance it
def enhance(input_path):
    os.makedirs('enhanced_srt_files', exist_ok=True)

    filename = os.path.basename(input_path)
    base, _ = os.path.splitext(filename)
    safe_base = safe_filename(base)
    output_path = os.path.join('enhanced_srt_files', f"{safe_base}.srt")

    print(f"Processing: {filename}")
    try:
        split_srt_blocks(input_path, output_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
