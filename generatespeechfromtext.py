from elevenlabs import ElevenLabs
import wave, time, os

client = ElevenLabs(api_key="sk_4a6dc7d6e75030ef5057b6facf4b5662123997762526cba3")
VOICE_ID = "kb9LZZlhckjFQsP89t9T"
OUTPUT_DIR = "dataset"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
os.makedirs(WAV_DIR, exist_ok=True)

def chunk_text(path):
    with open(path, "r") as f:
        text = f.read()
    
    # Split on sentence-ending punctuation, keeping it short
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # If adding this sentence keeps us under 200 chars, add it
        if len(current) + len(sentence) < 200:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    
    return chunks

chunks = chunk_text("script.txt")
print(f"Total chunks: {len(chunks)}")

metadata = []
for i, chunk in enumerate(chunks):
    # Skip very short chunks
    if len(chunk) < 20:
        continue

    fname = f"output_{i:04d}"
    wav_path = os.path.join(WAV_DIR, f"{fname}.wav")

    # Skip if already generated
    if os.path.exists(wav_path):
        print(f"[{i+1}/{len(chunks)}] Skipping {fname} (already exists)")
        metadata.append([fname, chunk])
        continue

    audio = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=chunk,
        model_id="eleven_turbo_v2_5",
        output_format="pcm_22050"
    )

    pcm_data = b"".join(audio)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(pcm_data)

    metadata.append([fname, chunk])
    print(f"[{i+1}/{len(chunks)}] {fname} — {len(chunk)} chars")
    time.sleep(0.5)

with open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w") as f:
    for fname, text in metadata:
        f.write(f"{fname}|{text}\n")

print(f"\nDone. {len(metadata)} utterances written to {OUTPUT_DIR}/")