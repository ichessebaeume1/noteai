from queue import Queue
import os
import sounddevice as sd
from scipy.io.wavfile import write
from config import config

class AudioStream:
    def __init__(self, audio_buffer, fs=44100, is_active=False, rec_history_save=5):
        self.audio_buffer = audio_buffer  # duration of each recording in seconds
        self.fs = fs  # Sample rate
        self.is_active = is_active
        self.rec_n = 1
        self.rec_history_save = rec_history_save
        self.dir = config.audio_output_path  # output directory
        self.queue = Queue()  # Queue for transcriber

    def start_stream(self):
        """Start the audio recording process."""
        self.is_active = True
        self._record_loop()  # Initiates the continuous recording loop

    def _record_loop(self):
        """Records audio in chunks and saves it continuously until stopped."""
        while self.is_active:
            recording = sd.rec(int(self.audio_buffer * self.fs), samplerate=self.fs, channels=1)
            sd.wait()  # Wait until recording is finished
            self._save_recording(recording)

    def _save_recording(self, recording):
        """Saves each recording as a WAV file and adds it to the transcription queue."""
        filename = f'{self.dir}/output_{self.rec_n}.wav'
        write(filename, self.fs, recording)  # Save recording to WAV file
        self.queue.put(filename)  # Add filename to the transcription queue
        self.rec_n += 1

    def stop_stream(self):
        """Stops the audio recording process."""
        self.is_active = False
        self.delete_all_audio()
        # Clear the queue when stopped
        while not self.queue.empty():
            self.queue.get()

    def delete_all_audio(self):
        for filename in os.listdir(self.dir):
            if os.path.isfile(os.path.join(self.dir, filename)):
                os.remove(os.path.join(self.dir, filename))
