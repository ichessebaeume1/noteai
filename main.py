import torch.cuda
from src.transcription.transcriber import Transcriber
from src.summarization.filters import Filter
from src.summarization.summarizer import Summarizer
from config import config

def main():
    print("Using CUDA for AI") if torch.cuda.is_available() else print("Using CPU for AI")
    print("Initializing audio, transcription, and summary components...")

    # Initialize audio, transcription, and summary components
    transcriber = Transcriber()
    summarizer = Summarizer(config.topic)
    batch_filter = Filter(config.topic)

    print("Starting recording...")
    transcription = transcriber.transcribe_audio_files()
    relevant_trans = batch_filter.process_and_filter_batch(transcription)

    if relevant_trans:
        summarizer.summarize_chunks(relevant_trans)
    else:
        print("No relevant transcripts found")


if __name__ == "__main__":
    main()
