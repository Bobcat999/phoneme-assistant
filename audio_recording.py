import librosa
import IPython.display as ipd
import sounddevice as sd
import numpy as np
import wave
from grapheme_to_phoneme import grapheme_to_phoneme
from process_audio import process_audio_array

def record_audio(filename, duration=5, fs=44100):
    """Records audio from the microphone and saves it to a WAV file.

    Args:
        filename (str): The name of the file to save the audio to.
        duration (int): The duration of the recording in seconds.
        fs (int): The sampling rate (samples per second).
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    print("Finished recording.")

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes because of np.int16
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

#Combining everything into a single function
def record_and_process_pronunciation(text, phoneme_extraction_model, use_previous_recording=False, word_extravtion_model=None):
    if not use_previous_recording:
        record_audio("output.wav")
    ground_truth_phonemes = grapheme_to_phoneme(text)

    y, sr = librosa.load("output.wav")
    output = process_audio_array(ground_truth_phonemes=ground_truth_phonemes, audio_array=y, sampling_rate=16000, phoneme_extraction_model=phoneme_extraction_model, word_extraction_model=word_extravtion_model)
    return output, ground_truth_phonemes

if __name__ == "__main__":
    from phoneme_extractor import PhonemeExtractor
    extractor = PhonemeExtractor()
    scentence = "The cat is on the mat"
    print(f"Say: {scentence}")
    output, ground_truth_phonemes = record_and_process_pronunciation(scentence, extractor)
    print(f"Output: {output}")
    print(f"Ground truth: {ground_truth_phonemes}")

    #GPT stuff
    print(f'Attempted scentence: {scentence}')
    print(f'Ground truth phonemes: {ground_truth_phonemes}')