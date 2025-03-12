from phoneme_extractor import PhonemeExtractor
import numpy as np
import IPython.display as ipd
import re
import accuracy_metrics as am
from grapheme_to_phoneme import grapheme_to_phoneme as g2p

def process_audio_array_verbose(audio_array, sampling_rate=16000, extraction_model=None, sample=None):
    """
    Use the phoneme extractor to transcribe an audio array.
    @returns: phoneme error rate, accuracy, ground truth text, transcription
    """
    if extraction_model is None:
        extraction_model = PhonemeExtractor()
    
    ipd.display(ipd.Audio(data=audio_array, rate=sampling_rate))

    transcription, logits, predicted_ids, top2_probs, top2_ids = extraction_model.extract_phoneme(audio=audio_array, sampling_rate=sampling_rate)

    print("model name", extraction_model.model_name, "\n")

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

    
    
def default_model_output_processing(transcription):
    # Filter out our transcription
    filtered_transcription = transcription[0]
    chars_to_remove = "ː0123"
    for char in chars_to_remove:
        filtered_transcription = filtered_transcription.replace(char, "")
    return filtered_transcription


def process_audio_array(ground_truth_phonemes, audio_array, sampling_rate=16000, extraction_model=None, model_output_filtering=default_model_output_processing):
    """
    Use the phoneme extractor to transcribe an audio array.
    @returns: phoneme error rate, accuracy, ground truth text, transcription
    """
    if extraction_model is None:
        extraction_model = PhonemeExtractor()
    
    if len(ground_truth_phonemes) <= 1:
        raise ValueError("ground_truth_phonemes must have at least 2 elements)")

    
    # get information from extraction
    transcription, logits, predicted_ids, top2_probs, top2_ids = extraction_model.extract_phoneme(audio=audio_array, sampling_rate=sampling_rate)

    filtered_transcription = model_output_filtering(transcription).replace("-", " ").split(" ")

    per = am.compute_phoneme_error_rate(ground_truth_phonemes, filtered_transcription)

    return transcription, per, filtered_transcription