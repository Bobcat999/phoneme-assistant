#%%
from dataset_loader import SpeechDatasetLoader
from grapheme_to_phoneme import grapheme_to_phoneme as g2p
from phonecodes.phonecodes import ipa2arpabet, arpabet2ipa
import numpy as np
import IPython.display as ipd
import accuracy_metrics as am
import re

ds = SpeechDatasetLoader()
#%%
from phoneme_extractor import PhonemeExtractor
extractor = PhonemeExtractor()
# %%

def process_index(index):
    # Transcribe the audio file
    sample = ds.train[index]
    audio_array = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]

    ipd.display(ipd.Audio(data=audio_array, rate=sampling_rate))

    transcription, logits, predicted_ids, top2_probs, top2_ids = extractor.extract_phoneme(audio=audio_array, sampling_rate=sampling_rate)

    print("model name", extractor.model_name, "\n")

    print("Ground truth text:", sample["text"])
    print("Original Transcription:", transcription)
    filtered_transcription = transcription[0]
    chars_to_remove = "ː0123"
    for char in chars_to_remove:
        filtered_transcription = filtered_transcription.replace(char, "")

    print("\nTranscription modified:",filtered_transcription)
    print("Ground truth arpa:", " ".join(
        ["-".join(word["phones"]) for word in sample["words"]]
    ))
    ground_truth_phonemes = am.normalize_phonemes([ph for word in sample["words"] for ph in word["phones"]])
    predicted_phonemes = am.normalize_phonemes(re.split(r" |-", filtered_transcription))
    print("\nGround truth phonemes:", ground_truth_phonemes)
    print("Predicted phonemes:", predicted_phonemes)
    per = am.compute_phoneme_error_rate(ground_truth_phonemes, predicted_phonemes)
    print(f"Phoneme Error Rate: {per:.2%}")
    print(f"Ground truth accuracy: {sample['accuracy']/10.0:.2%}")

    # print("Ground truth phoneme", [word["phones"] for word in sample["words"]])
    print("\nTranslated grapheme", "".join(g2p(sample["text"].lower())))

    return per, sample["accuracy"]/10.0, sample["text"], transcription

# %%
import pandas as pd

df = pd.DataFrame(columns=["PER", "Accuracy", "Text", "Transcription"])

avg_per = 0


for i in range(0,10):
    df.loc[i] = process_index(i)

#%%
print(df)

#%%
process_index(0)