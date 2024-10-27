from io import BytesIO

import streamlit as st

from utils import (
    check_audio_format,
    check_video_format,
    display_sentiment_scores,
    display_sentiment_summary,
    display_typing_effect,
    extract_audio_from_video,
    get_sentiment,
    load_audio,
    normalize_audio,
    record_audio,
    speech_to_text,
    textual_sentiment_analysis,
    tonal_sentiment_analysis,
)


def main():
    st.title("Sentiment Analysis on Audio")

    with st.sidebar:
        input_audio_file = st.file_uploader("Upload an audio file:", type=["wav", "mp3"])
        input_video_file = st.file_uploader("Upload a video file:", type=["mp4", "mov"])

    # Process audio or video file if uploaded
    if input_audio_file is not None:
        process_file(input_audio_file, is_audio=True)
    elif input_video_file is not None:
        process_file(input_video_file, is_audio=False)


def process_file(file, is_audio=True):
    # Set up buffer and check format
    file_name = file.name
    file_bytes = file.read()
    file_buffer = BytesIO(file_bytes)

    if is_audio:
        check_audio_format(file_name)
        audio, _ = load_audio(file_buffer)
    else:
        check_video_format(file_name)
        audio, _ = extract_audio_from_video(file_buffer)

    # Run speech to text
    transcription = run_transcription(audio)

    # Run sentiment analysis
    text_sentiment_scores = analyze_text_sentiment(transcription)
    tone_sentiment_scores = analyze_tone_sentiment(audio)

    # Display summary
    display_summary(transcription, text_sentiment_scores, tone_sentiment_scores)


def run_transcription(audio):
    transcription = speech_to_text(audio)
    st.subheader("Transcription:")
    typing_placeholder = st.empty()  # placeholder for dynamic content
    display_typing_effect(transcription, typing_placeholder)
    return transcription


def analyze_text_sentiment(transcription):
    text_sentiment_scores = textual_sentiment_analysis(transcription)
    st.subheader("Sentiment Analysis on Text:")
    display_sentiment_scores(text_sentiment_scores)
    return text_sentiment_scores


def analyze_tone_sentiment(audio):
    tone_sentiment_scores = tonal_sentiment_analysis(audio)
    st.subheader("Sentiment Analysis on Tone:")
    display_sentiment_scores(tone_sentiment_scores)
    return tone_sentiment_scores


def display_summary(transcription, text_sentiment_scores, tone_sentiment_scores):
    text_sentiment, tone_sentiment = get_sentiment(text_sentiment_scores, tone_sentiment_scores)
    st.subheader("Summary")
    typing_placeholder = st.empty()
    display_sentiment_summary(text_sentiment, tone_sentiment, typing_placeholder)

if __name__ == "__main__":
    main()