import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from st_audiorec import st_audiorec
from utils import upload_file_and_preprosses , qa_retrieval 
load_dotenv()
from streamlit_mic_recorder import speech_to_text
import boto3
from contextlib import closing

API_KEY = os.getenv("OPENAI_API_KEY")
st.markdown("""
    <style>
    .title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #4B4B4B;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
        background: -webkit-linear-gradient(#f8cdda, #1e90ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #4B4B4B;
        font-size: 1.5em;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Display the styled title
st.markdown('<div class="title">VoiceQuery AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speech-to-Text and Text-to-Speech Assistant</div>', unsafe_allow_html=True)
# Display description
st.markdown("""
    <style>
    .description {
        font-family: 'Helvetica Neue', sans-serif;
        color: #007ACC;
        font-size: 1em;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="description">Upload a document file and click on the Submit button. After processing, choose your query method: Text or Audio. You can also play the response in audio form.</div>', unsafe_allow_html=True)

result = upload_file_and_preprosses()

# Transcribe voice to text (if needed)
def transcribe_voice_to_text(audio_location):
    client = OpenAI(api_key=API_KEY)
    with open(audio_location, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

def chat_completion_call(text):
    result = qa_retrieval(text)
    return result

# Generate chat completion
# def chat_completion_call(text):
#     client = OpenAI(api_key=API_KEY)
#     messages = [{"role": "user", "content": text}]
#     response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
#     return response.choices[0].message.content

def text_to_speech_ai(response_text):
    polly = boto3.client(
        'polly',
        region_name='us-east-1',
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key")
    )

    try:
        # Synthesize speech
        polly_response = polly.synthesize_speech(Text=response_text, OutputFormat="mp3", VoiceId="Aditi")

        # Log response metadata for debugging
        print("Polly Response Metadata:", polly_response.get('ResponseMetadata', {}))

        # Check for audio stream in Polly response
        if "AudioStream" in polly_response:
            with closing(polly_response["AudioStream"]) as stream:
                audio_content = stream.read()
                # Save the MP3 file
                with open("output.mp3", "wb") as file:
                    file.write(audio_content)
                print("Audio saved to output.mp3")
                return audio_content
        else:
            print("AudioStream not found in Polly response.")
            return None
    except Exception as error:
        print(f"Error synthesizing speech: {error}")
        return None
# Convert text to speech
# def text_to_speech_ai(response):
#     client = OpenAI(api_key=API_KEY)
#     tts_response = client.audio.speech.create(model="tts-1-hd", voice="nova", input=response)
#     audio_content = tts_response.content  # Get audio content as bytes
#     return audio_content


query_method = st.radio("Select Query Method", ("Text", "Audio"))
if query_method == "Text":
    # Initialize session state
    if "ai_response" not in st.session_state:
        st.session_state.ai_response = ""
    if "audio_content" not in st.session_state:
        st.session_state.audio_content = None
    # Get user query
    input_text = st.text_input("Enter your Query here ...")
    audio_file_path = "ai_response_audio.mp3"
    if st.button("Submit"):
        # Get AI response in text format
        ai_response_text = chat_completion_call(input_text)
        st.write("AI Response: ", ai_response_text)
        st.session_state.ai_response = ai_response_text
        st.session_state.audio_content = ""

    # Show Play button if AI response is available
    if st.session_state.ai_response:
        if st.button("Play AI Response Audio"):
            # Generate audio content for the AI response only when the button is clicked
            if not st.session_state.audio_content:
                st.session_state.audio_content = text_to_speech_ai(st.session_state.ai_response)
            st.write("AI Response: ", st.session_state.ai_response)
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(st.session_state.audio_content)
            st.audio(audio_file_path, format='audio/mp3', start_time=0)


if query_method == "Audio":
    # Initialize session state for audio response
    if "ai_response_text_audio" not in st.session_state:
        st.session_state.ai_response_text_audio = ""
    if "audio_content_audio" not in st.session_state:
        st.session_state.audio_content_audio = None
    
    # Use speech-to-text for audio input
    text = speech_to_text(use_container_width=True, just_once=True, key="STT")
    
    if text:
        st.write(f"Transcribed Text: {text}")
        # Get AI response from transcribed text
        ai_response_text_audio = chat_completion_call(text)
        st.write("AI Response: ", ai_response_text_audio)
        st.session_state.ai_response_text_audio = ai_response_text_audio
        st.session_state.audio_content_audio = None  # Reset audio content for new query
    
    # Show Play button if AI response is available
    if st.session_state.ai_response_text_audio:
        if st.button("Play AI Response Audio (Audio Query)"):
            # Generate audio content for the AI response only when the button is clicked
            if not st.session_state.audio_content_audio:
                st.session_state.audio_content_audio = text_to_speech_ai(st.session_state.ai_response_text_audio)
            # Save and play the audio response
            audio_file_path = "ai_response_audio.mp3"
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(st.session_state.audio_content_audio)
            st.audio(audio_file_path, format='audio/mp3', start_time=0)

    # Show Play button if AI response is available
    # if response:
    #     if st.button("Play AI Response Audio"):
    #         # Generate audio content for the AI response only when the button is clicked
    #         if not aud_content:
    #             aud_content = text_to_speech_ai(response)
    #         audio_file_path = "ai_response_audio.mp3"
    #         with open(audio_file_path, "wb") as audio_file:
    #             audio_file.write(aud_content)
    #         st.audio(audio_file_path, format='audio/mp3', start_time=0)
