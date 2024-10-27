import mimetypes
import time

import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import streamlit as st
import torch
from pydub import AudioSegment
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)


def check_audio_format(audio_file):
    """
    Checks if the provided file is a valid audio file based on its MIME type.

    Parameters:
        audio_file (str): The path to the audio file to be checked.

    Raises:
        ValueError: If the file is not a valid audio file.
    """
    mime_type, _ = mimetypes.guess_type(audio_file)
    if not mime_type or not mime_type.startswith('audio'):
        raise ValueError("The input file is not a valid audio file")
    

def load_audio(audio_file):
    """
    Load an audio file using librosa with a sampling rate of 16kHz.

    Parameters:
    audio_file (str): Path to the audio file to be loaded.

    Returns:
    tuple: A tuple containing the audio time series and the sampling rate.
    """
    audio, sample_rate = librosa.load(audio_file, sr=16000) 
    
    return audio, sample_rate


def speech_to_text(audio):
    """
    Converts speech audio to text using the Wav2Vec2 model.
    Args:
        audio (numpy.ndarray): The input audio signal to be transcribed.
    Returns:
        str: The transcribed text from the audio input.
    """
    # Instantiate processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Process audio    
    processed_audio = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(processed_audio).logits

    # Decode the predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription
    

def textual_sentiment_analysis(text):
    """
    Analyzes the sentiment of the given text using a pre-trained RoBERTa model.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary with sentiment labels ('Negative', 'Neutral', 'Positive') as keys and their corresponding scores as values.
    """
    # Instantiate tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        sentiment = model(**tokens)

    # Get scores
    scores = sentiment[0][0].detach().numpy()
    scores = softmax(scores)
    ids = ['Negative', 'Neutral', 'Positive']
    id_to_score = dict(zip(ids, map(lambda x: round(float(x), 4), scores)))

    return id_to_score


def record_audio(duration=5, sample_rate=16000):
    """
    Records audio from the default microphone for a specified duration and sample rate.

    Parameters:
    duration (int, optional): The length of the recording in seconds. Default is 5 seconds.
    sample_rate (int, optional): The sample rate for the recording in Hertz. Default is 16000 Hz.

    Returns:
    numpy.ndarray: A 1D numpy array containing the recorded audio data.
    None: If an error occurs during recording.

    Raises:
    Exception: If an error occurs during the recording process, it will be caught and printed.
    """
    try:
        print("Recording... Speak now!")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        return np.squeeze(audio)
    except Exception as e:
        print("Error recording audio:", e)
        return None
    

def normalize_audio(audio):
    """
    Normalize the audio signal to have values between -1 and 1.

    Parameters:
    audio (numpy.ndarray): The input audio signal as a numpy array.

    Returns:
    numpy.ndarray: The normalized audio signal. If the input audio is None or 
                   has no non-zero values, the original audio is returned.
    """
    if audio is not None and np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def check_video_format(video_file):
    """
    Checks if the provided file is a valid video file based on its MIME type.

    Parameters:
        video_file (str): The path to the video file to be checked.

    Raises:
        ValueError: If the file is not a valid video file.
    """
    mime_type, _ = mimetypes.guess_type(video_file)
    if not mime_type or not mime_type.startswith('video'):
        raise ValueError("The input file is not a valid video file")
    

def extract_audio_from_video(video_file, sample_rate=16000):
    """
    Extracts audio from a video file, resamples it to the specified sample rate, 
    converts it to mono, and normalizes the audio.
    Args:
        video_file (str): Path to the video file from which to extract audio.
        sample_rate (int, optional): Target sample rate for the audio. Defaults to 16000.
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The normalized audio data as a numpy array.
            - int: The sample rate of the audio.
    """
    # Load the audio segment from the video file
    audio = AudioSegment.from_file(video_file, format="mp4")
    
    # Resample audio to the target sample rate
    audio = audio.set_frame_rate(sample_rate).set_channels(1)  # Convert to mono
    
    # Convert audio to numpy array
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize audio
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array, sample_rate


def tonal_sentiment_analysis(audio):
    """
    Perform tonal sentiment analysis on an audio input using a pre-trained Wav2Vec2 model.
    Args:
        audio (numpy.ndarray): The input audio waveform.
    Returns:
        dict: A dictionary mapping sentiment labels ('Neutral', 'Happy', 'Angry', 'Sad') to their respective scores.
    """
    # Instantiate feature extractor and model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")

    # Compute attention masks and normalize the waveform if needed
    features = feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        sentiment = model(**features)
    
    # Get scores
    scores = sentiment[0][0].detach().numpy()
    scores = softmax(scores)
    ids = ['Neutral', 'Happy', 'Angry', 'Sad']
    id_to_score = dict(zip(ids, map(lambda x: round(float(x), 4), scores)))

    return id_to_score


def display_typing_effect(text, placeholder, delay=0.1):
    """
    Displays text in a Streamlit app with a typing effect, word by word.
    
    Parameters:
    - text (str): The text to display with the typing effect.
    - placeholder (st.delta_generator.DeltaGenerator): A Streamlit placeholder for dynamic content.
    - delay (float): Time delay in seconds between each word.
    """
    displayed_text = ""
    for word in text.split():
        displayed_text += word.lower() + " "
        placeholder.markdown(
            f"""
            <div style="
                background-color: #f1f1f1;
                padding: 10px 15px;
                border-radius: 15px;
                width: fit-content;
                max-width: 90%;
                margin-bottom: 20px;
                font-size: 16px;
                color: #333;
            ">
                {displayed_text}
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(delay)


def display_sentiment_scores(sentiment_scores):
    """
    Displays sentiment scores in a table format using Streamlit.
    
    Parameters:
    - sentiment_scores (dict): Dictionary with sentiment names as keys and scores as values.
    """
    # Convert the dictionary to a DataFrame for table display
    sentiment_df = pd.DataFrame(list(sentiment_scores.items()), columns=['Sentiment', 'Score'])
    
    # Display the DataFrame as a table
    st.table(sentiment_df)


def get_sentiment(text_sentiment_scores, tone_sentiment_scores):
    """
    Determine the predominant sentiment from text and tone sentiment scores.

    Args:
        text_sentiment_scores (dict): A dictionary where keys are sentiment labels and values are their corresponding scores for text.
        tone_sentiment_scores (dict): A dictionary where keys are sentiment labels and values are their corresponding scores for tone.

    Returns:
        tuple: A tuple containing the predominant sentiment label for text and the predominant sentiment label for tone.
    """
    text_sentiment = max(text_sentiment_scores, key=text_sentiment_scores.get)
    tone_sentiment = max(tone_sentiment_scores, key=tone_sentiment_scores.get)

    return text_sentiment, tone_sentiment


def display_sentiment_summary(text_sentiment, tone_sentiment, placeholder, delay=0.1):
    """
    Displays text in a Streamlit app with a typing effect, keeping all parts in the same line.

    Parameters:
    - text_sentiment (str): Sentiment derived from text analysis.
    - tone_sentiment (str): Sentiment derived from tone analysis.
    - placeholder (st.delta_generator.DeltaGenerator): A Streamlit placeholder for dynamic content.
    - delay (float): Time delay in seconds between each part.
    """
    
    # Prepare text to display inline
    cumulative_text = (
        "From the words he is speaking, the individual seems to have a "
        + text_sentiment.lower()
        + " emotion. From his tone, he seems to be "
        + tone_sentiment.lower()
        + "."
    )
    
    # Display each word with typing effect in the same line
    displayed_text = ""
    for word in cumulative_text.split():
        displayed_text += word + " "
        placeholder.markdown(
            f"""
            <div style="
                background-color: #f1f1f1;
                padding: 10px 15px;
                border-radius: 15px;
                width: fit-content;
                max-width: 90%;
                margin-bottom: 20px;
                font-size: 16px;
                color: #333;
            ">
                {displayed_text.strip()}
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(delay)