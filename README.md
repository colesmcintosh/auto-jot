# AutoJot

AutoJot is an application that automates note-taking by transcribing audio files and generating insightful notes from the transcriptions. It uses the OpenAI Whisper ASR API to transcribe the audio and the GPT-4 model to generate notes, including summaries, key points, and action items.

Live demo: https://autojot.onrender.com/

> **Note:** The live demo may be slow to load due to the free hosting service. If you experience any issues, please try running the app locally.

## Features

- Supports multiple audio formats (mp3, mp4, mpeg, mpga, m4a, wav, webm)
- Automatically transcribes audio files using the OpenAI Whisper ASR API
- Generates notes from transcriptions using the GPT-4 model
- Splits large audio files into smaller chunks for easy transcription
- Automatically handles context length exceeded issues by truncating the text

## Installation

1. Clone this repository or download the code files
2. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key in the code:

   ```python
   openai.api_key = "your_openai_api_key_here"
   ```

4. Run the Streamlit app:

   ```
   streamlit run app.py
   ```

## Usage

1. Launch the AutoJot app by running the Streamlit command
2. Upload an audio file in one of the supported formats
3. Click the "Generate Notes" button to start the transcription and note generation process
4. View the generated transcript and notes in the app
