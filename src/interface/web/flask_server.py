from flask import Flask, request, jsonify, render_template
from src.summarization.filters import Filter
from src.summarization.summarizer import Summarizer
from config import config

app = Flask(__name__)

# Initialize filter and summarizer components
filter = Filter()
summarizer = Summarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-transcription', methods=['POST'])
def process_transcription():
    data = request.get_json()
    transcription = data.get('transcription', '')
    topic = data.get('topic', config.topic)  # Use the provided topic or default to the config topic

    # Check if transcription or topic is missing or empty
    if not transcription or not topic:
        return jsonify({"summary": "No valid transcription or topic provided."})

    # Filter and summarize the transcription
    filtered_text = filter.process_and_filter_batch(transcription, topic)
    if filtered_text:
        summary = summarizer.summarize_chunks(filtered_text, topic)
        return jsonify({"summary": summary})
    else:
        return jsonify({"summary": "No relevant content detected."})


if __name__ == '__main__':
    app.run(debug=True)
