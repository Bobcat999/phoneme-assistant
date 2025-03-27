import audio_recording
import phoneme_extractor

try:
    extractor = phoneme_extractor.PhonemeExtractor() 
    print(audio_recording.record_and_process_pronunciation("the quick brown fox jumped over the lazy dog", phoneme_extraction_model=extractor))
except Exception as e:
    print("An exception occurred")
    print(e)

