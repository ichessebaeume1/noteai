import whisper
from config import config
from src.summarization.filters import Filter  # Import the Filter class
from src.summarization.summarizer import Summarizer
import torch

class Transcriber:
    def __init__(self):
        self.model = config.gpu_model if torch.cuda.is_available() else config.cpu_model
        self.filter = Filter(config.topic)
        self.summarizer = Summarizer(config.topic)

    def transcribe_audio_files(self, audio_filename):
        result = self.get_transcription(audio_filename)

        filtered_batch = self.filter.process_and_filter_batch(result)
        if filtered_batch:
            print(filtered_batch)
            return self.summarizer.summarize_chunks(filtered_batch)
        return None

    def get_transcription(self, audio_path):
        model = whisper.load_model(self.model)
        result = model.transcribe(audio_path)
        return result["text"]
