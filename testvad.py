import sounddevice as sd
import numpy as np
import webrtcvad
import threading
import queue
import wave
import time

def run_vad(fs=16000, chunk_duration=0.03, vad_mode=3, silence_threshold=50, output_file="output.wav"):
    chunk_size = int(fs * chunk_duration)
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    def record_audio():
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16',
                            blocksize=chunk_size, callback=audio_callback):
            print("Recording started...")
            while not stop_event.is_set():
                time.sleep(0.1)
            print("Recording stopped.")

    def process_vad():
        vad = webrtcvad.Vad(vad_mode)
        silent_chunks = 0
        frames = []

        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            is_speech = vad.is_speech(chunk.tobytes(), fs)
            frames.append(chunk)

            if not is_speech:
                silent_chunks += 1
                if silent_chunks > silence_threshold:
                    stop_event.set()
            else:
                silent_chunks = 0

        # Save recorded audio to a WAV file
        audio_data = np.concatenate(frames)
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio saved to {output_file}")

    # Start threads
    recording_thread = threading.Thread(target=record_audio)
    vad_thread = threading.Thread(target=process_vad)

    recording_thread.start()
    vad_thread.start()

    recording_thread.join()
    vad_thread.join()

if __name__ == "__main__":
    run_vad()