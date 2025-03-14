from audio_processing.phoneme_extractor import PhonemeExtractor
import numpy as np
import IPython.display as ipd
import re
import evaluation.accuracy_metrics as am
from audio_processing.grapheme_to_phoneme import grapheme_to_phoneme as g2p

def process_audio_array_verbose(audio_array, sampling_rate=16000, extraction_model=None, sample=None):
    """
    Use the phoneme extractor to transcribe an audio array.
    @returns: phoneme error rate, accuracy, ground truth text, transcription
    """
    ground_truth_phonemes = am.normalize_phonemes([ph for word in sample["words"] for ph in word["phones"]])
    transcription, per = process_audio_array(ground_truth_phonemes, audio_array, sampling_rate, extraction_model=extraction_model)
    
    ipd.display(ipd.Audio(data=audio_array, rate=sampling_rate))

    print("model name", extraction_model.model_name, "\n")

    print("Ground truth text:", sample["text"])
    print("Transcription:", transcription)

    print("\nGround truth phonemes:", ground_truth_phonemes)
    print("Predicted phonemes:", transcription)
    print(f"Phoneme Error Rate: {per:.2%}")
    print(f"Ground truth accuracy: {sample['accuracy']/10.0:.2%}")

    # print("Ground truth phoneme", [word["phones"] for word in sample["words"]])
    print("\nTranslated grapheme", g2p(sample["text"].lower()))

    return transcription, per, sample["accuracy"]/10.0, sample["text"]

    
    


def process_audio_array(ground_truth_phonemes, audio_array, sampling_rate=16000, extraction_model=None):
    """
    Use the phoneme extractor to transcribe an audio array.
    @returns: phoneme error rate, accuracy, ground truth text, transcription
    """
    if extraction_model is None:
        extraction_model = PhonemeExtractor()
    
    if len(ground_truth_phonemes) <= 1:
        raise ValueError("ground_truth_phonemes must have at least 2 elements)")

    
    # get information from extraction
    transcription = extraction_model.extract_phoneme(audio=audio_array, sampling_rate=sampling_rate)

    per = am.compute_phoneme_error_rate(ground_truth_phonemes, transcription)

    return transcription, per