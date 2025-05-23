import os
import re
from datetime import datetime, timedelta


def clean_text_lines(text_lines):
    cleaned_lines = []
    for line in text_lines:
        line_clean = re.sub(r'\[.*?\]', '', line).strip()
        cleaned_lines.append(line_clean)
    return cleaned_lines

def parse_time(s):
    return datetime.strptime(s, "%H:%M:%S,%f")

def format_time(t):
    return t.strftime("%H:%M:%S,%f")[:-3]

def split_time(start, end):
    midpoint = start + (end - start) / 2
    return (start, midpoint), (midpoint, end)

def split_text_lines(lines):
    mid = len(lines) // 2
    return lines[:mid], lines[mid:]

def split_srt_blocks(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    temp_blocks = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        time_line = lines[1]
        text_lines = clean_text_lines(lines[2:])

        times = time_line.split(' --> ')
        try:
            start = parse_time(times[0].strip())
            end = parse_time(times[1].strip())
        except Exception as e:
            print(f"Skipping invalid time format in {input_path}: {time_line}")
            continue

        if len(text_lines) > 1:
            (start1, end1), (start2, end2) = split_time(start, end)
            part1, part2 = split_text_lines(text_lines)

            temp_blocks.append((start1, end1, part1))
            temp_blocks.append((start2, end2, part2))
        else:
            temp_blocks.append((start, end, text_lines))

    # Merge identical consecutive text blocks
    merged_blocks = []
    for start, end, lines in temp_blocks:
        if not merged_blocks:
            merged_blocks.append((start, end, lines))
        else:
            prev_start, prev_end, prev_lines = merged_blocks[-1]
            if lines == prev_lines:
                merged_blocks[-1] = (prev_start, end, prev_lines)
            else:
                merged_blocks.append((start, end, lines))

    # Write output with proper indexing
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (start, end, lines) in enumerate(merged_blocks, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n\n")

def safe_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', '_', name)

def batch_split_srts(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".srt"):
            input_path = os.path.join(input_folder, filename)
            base, _ = os.path.splitext(filename)
            safe_base = safe_filename(base)
            output_path = os.path.join(output_folder, f"{safe_base}.clean.srt")
            print(f"Processing: {filename}")
            try:
                split_srt_blocks(input_path, output_path)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

# Utilisation :
batch_split_srts("srt_files", "enhanced_srt_files")
