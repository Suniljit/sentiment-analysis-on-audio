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


    if input_audio_file is not None:
        
        # Load and process audio input
        audio_file = input_audio_file.name
        audio_bytes = input_audio_file.read()
        audio_buffer = BytesIO(audio_bytes)  # Create an in-memory buffer
        check_audio_format(audio_file)
        audio, _ = load_audio(audio_buffer)

        # Perform speech to text
        transcription = speech_to_text(audio)

        # Display transcription
        st.subheader("Transcription:")
        typing_placeholder = st.empty() # placeholder for dynamic content
        display_typing_effect(transcription, typing_placeholder)
        
        # Perform sentiment analysis on text
        text_sentiment_scores = textual_sentiment_analysis(transcription)

        # Display sentiment scores
        st.subheader("Sentiment Analysis on Text:")
        display_sentiment_scores(text_sentiment_scores)
        
        # Perform sentiment analysis on tone
        tone_sentiment_scores = tonal_sentiment_analysis(audio)

        # Display sentiment scores
        st.subheader("Sentiment Analysis on Tone:")
        display_sentiment_scores(tone_sentiment_scores)

        # Display Summary
        text_sentiment, tone_sentiment = get_sentiment(text_sentiment_scores, tone_sentiment_scores)
        st.subheader("Summary")
        typing_placeholder = st.empty()
        display_sentiment_summary(text_sentiment, tone_sentiment, typing_placeholder)

    if input_video_file is not None:
        
        # Load and process audio input
        video_file = input_video_file.name
        video_bytes = input_video_file.read()
        video_buffer = BytesIO(video_bytes)  # Create an in-memory buffer
        check_video_format(video_file)
        audio, _ = extract_audio_from_video(video_buffer)

        # Perform speech to text
        transcription = speech_to_text(audio)

        # Display transcription
        st.subheader("Transcription:")
        typing_placeholder = st.empty() # placeholder for dynamic content
        display_typing_effect(transcription, typing_placeholder)
        
        # Perform sentiment analysis on text
        text_sentiment_scores = textual_sentiment_analysis(transcription)

        # Display sentiment scores
        st.subheader("Sentiment Analysis on Text:")
        display_sentiment_scores(text_sentiment_scores)
        
        # Perform sentiment analysis on tone
        tone_sentiment_scores = tonal_sentiment_analysis(audio)

        # Display sentiment scores
        st.subheader("Sentiment Analysis on Tone:")
        display_sentiment_scores(tone_sentiment_scores)

        # Display Summary
        text_sentiment, tone_sentiment = get_sentiment(text_sentiment_scores, tone_sentiment_scores)
        st.subheader("Summary")
        typing_placeholder = st.empty()
        display_sentiment_summary(text_sentiment, tone_sentiment, typing_placeholder)

if __name__ == "__main__":
    main()