import streamlit as st
import openai
import tempfile
import numpy as np
import audioread
import os
import scipy.io.wavfile
from transformers import GPT2Tokenizer
import time
from dotenv import load_dotenv

load_dotenv()

def truncate_text(text, max_tokens=8000):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(truncated_tokens)

# Set the OpenAI API key
openai.api_key = os.getenv("SECRET_KEY")

def transcribe_audio_chunk(audio_chunk):
    if len(audio_chunk) == 1:  # small file, just the file path
        file_path = audio_chunk[0]
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
    else:  # large file, split into chunks
        audio_data, rate, channels = audio_chunk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            scipy.io.wavfile.write(f.name, rate, audio_data)
            with open(f.name, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

def generate_notes(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in analyzing and explaining audio transcriptions provided by Whisper, an automatic speech recognition (ASR) system. Provide a detailed and clear explanation of the transcription. Create seperate sections for a summary, notes, key points, action items, and any important information.",
            },
            {
                "role": "user",
                "content": f"Transcription:\n{prompt}",
            },
        ],
    )
    return response.choices[0].message['content'].strip()

def split_large_audio_file(file_path, max_size_mb=20):
    with audioread.audio_open(file_path) as audio_file:
        duration = audio_file.duration * 1000  # convert to milliseconds
        size_bytes = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024

        if size_bytes > max_size_bytes:
            rate = audio_file.samplerate
            channels = audio_file.channels
            total_frames = int(rate * duration / 1000)
            bytes_per_frame = channels * 2  # 2 bytes per sample (16-bit audio)
            split_frames = int(max_size_bytes / bytes_per_frame)
            frame_data = []

            for frame in audio_file:
                frame_data.append(np.frombuffer(frame, dtype=np.int16))

            audio_data = np.concatenate(frame_data)[:total_frames]
            audio_chunks = []

            for start in range(0, total_frames, split_frames):
                end = min(start + split_frames, total_frames)
                chunk_data = audio_data[start:end]
                audio_chunks.append((chunk_data, rate, channels))

            return audio_chunks
        else:
            return [(file_path,)]  # return the original file path as a single-item list

st.set_page_config(
    page_title="AutoJot",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("AutoJot 📝")

# Allow multiple file formats for uploading
uploaded_file = st.file_uploader("Upload an audio file (mp3, mp4, mpeg, mpga, m4a, wav, or webm)", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as f:
        f.write(uploaded_file.getvalue())
        temp_file_path = f.name

    if st.button("Generate Notes"):
        try:
            with st.spinner("Processing your audio file"):
                audio_chunks = split_large_audio_file(temp_file_path)

            transcripts = []
            total_chunks = len(audio_chunks)
            progress_bar = st.progress(0)

            for i, chunk in enumerate(audio_chunks):
                with st.spinner(f"Transcribing part {i+1} of {total_chunks}"):
                    transcripts.append(transcribe_audio_chunk(chunk))
                    progress_bar.progress((i+1) / total_chunks)

            transcript = " ".join(transcripts)
            trans_success = st.success("Successfully transcribed", icon="✅")
            with st.expander("See transcription"):
                st.write(transcript)
            time.sleep(5)
            trans_success.empty()

            if transcript:
                try:
                    with st.spinner("Generating Notes..."):
                        notes = generate_notes(transcript)
                        note_success = st.success("Successfully generated notes", icon="✅")
                        st.write(notes)
                except Exception as e:
                    with st.spinner("Generating Notes (with truncation)..."):
                        truncated_transcript = truncate_text(transcript)
                        try:
                            notes = generate_notes(truncated_transcript)
                            note_success = st.success("Successfully generated notes", icon="✅")
                            st.write(notes)
                        except Exception as e2:
                            st.error(f"An error occurred: {e2}")
        except Exception as e:
                    st.error(f"An error occurred: {e}")
