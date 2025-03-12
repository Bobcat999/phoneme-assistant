import librosa
from process_audio import process_audio_array
from dataset_loader import SpeechDatasetLoader


# We want to compare some models to see which one is the best for our data
dataset = SpeechDatasetLoader()
train = dataset.train
sampling_rate = 16000

audio = train[0][""]

# Get the model
from phoneme_extractor import PhonemeExtractor
extractor = PhonemeExtractor()

# print(extractor.extract_phoneme(audio_array, sampling_rate))


# print(process_audio_array())