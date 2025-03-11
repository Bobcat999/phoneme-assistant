#%%
from dataset_loader import SpeechDatasetLoader
from grapheme_to_phoneme import grapheme_to_phoneme as g2p
from phonecodes.phonecodes import ipa2arpabet, arpabet2ipa
import librosa

ds = SpeechDatasetLoader()
#%%
from phoneme_extractor import PhonemeExtractor
extractor = PhonemeExtractor()
# %%
from process_audio import process_audio_array_verbose

def process_index(index, audio_file=None):
    # Transcribe the audio file
    sample = ds.train[index]
    if audio_file is not None:
        audio_array, sampling_rate = librosa.load(audio_file)
        sampling_rate = 16000
    else:
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
    
    process_audio_array_verbose(audio_array, sampling_rate, extraction_model=extractor, sample=sample)
    


# %%
import pandas as pd

df = pd.DataFrame(columns=["PER", "Accuracy", "Text", "Transcription"])

avg_per = 0


for i in range(400,410):
    df.loc[i] = process_index(i)

#%%
print(df)

#%%
process_index(1001)
# %%
