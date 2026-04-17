import os
import re

SCRIPT_FILE = "script.txt"
WAV_DIR = "dataset/wavs"
OUTPUT_CSV = "dataset/metadata.csv"

def chunk_text(path):
    with open(path, "r") as f:
        text = f.read()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) < 200:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

chunks = chunk_text(SCRIPT_FILE)

with open(OUTPUT_CSV, "w") as f:
    for i, chunk in enumerate(chunks):
        if len(chunk) < 20:
            continue
        fname = f"output_{i:04d}"
        wav_path = os.path.join(WAV_DIR, f"{fname}.wav")
        if os.path.exists(wav_path):
            clean_chunk = chunk.replace("\n", " ").replace("\r", " ")
            clean_chunk = " ".join(clean_chunk.split())
            f.write(f"{fname}|{clean_chunk}\n")

print(f"Done — wrote metadata.csv")
