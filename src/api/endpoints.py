from fastapi import FastAPI, UploadFile, File
from src.transcription.transcriber import Transcriber
from src.summarization.filters import Filter
from src.summarization.summarizer import Summarizer
from config import config
import tempfile

app = FastAPI()

# Initialize components
transcriber = Transcriber()
summarizer = Summarizer(config.topic)
filter_instance = Filter(config.topic)


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    # Decrypt and save the audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    # Perform transcription
    transcription = transcriber.get_transcription(temp_file_path)

    # Calculate relevance score and convert to a Python float
    relevance_score = float(filter_instance.assess_relevance_score(transcription))
    filtered_batch = filter_instance.process_and_filter_batch(transcription)

    if filtered_batch:
        summary = summarizer.summarize_chunks(filtered_batch)
    else:
        summary = "No relevant content found."

    return {
        "transcription": transcription,
        "filter_confidence": relevance_score,
        "summary": summary
    }


@app.post("/process_transcript/")
async def process_transcript(transcript: str):
    # Directly summarize the provided transcript
    summary = summarizer.summarize_chunks(transcript)
    return {"summary": summary}
