import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import re


def default_model_output_processing(transcription):
    # Filter out our transcription
    filtered_transcription = transcription[0]
    chars_to_remove = "ː0123"
    for char in chars_to_remove:
        filtered_transcription = filtered_transcription.replace(char, "")
    filtered_transcription = re.split(r" |-", filtered_transcription)
    filtered_transcription = [phoneme for phoneme in filtered_transcription if phoneme != ""]
    return filtered_transcription

class PhonemeExtractor:
    def __init__(self, model_name =  "mirfan899/kids_phoneme_sm_model", model_output_processing=default_model_output_processing):
        # Replace with your pre-trained phoneme model identifier from Hugging Face
        # self.model_name = "speech31/wav2vec2-large-english-TIMIT-phoneme_v3"
        self.model_name = model_name
        # Load the phoneme tokenizer and model
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_name,
            phonemizer_kwargs={'phoneme_format': 'arpabet'}
        )

        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)

        self.blank_token_id = self.processor.tokenizer.pad_token_id # for CTC loss

        self.model_output_processing = model_output_processing

    def collapse_repeats(self, pred_ids):
        """
        Collapse repeated tokens and remove blank tokens.
        For example, if pred_ids is [A, A, blank, B, B, C, C],
        it will return [A, B, C].
        """
        collapsed = []
        previous = None
        for token in pred_ids:
            # Skip blank tokens
            if token == self.blank_token_id:
                previous = token
                continue
            # Only add if different from previous token
            if token != previous:
                collapsed.append(token)
            previous = token
        return collapsed

    def extract_phoneme(self, audio, sampling_rate=16000):
        # Load the audio file
        # Tokenize the audio file
        input_values = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values
        # retrieve logits from the model
        with torch.no_grad():
            logits = self.model(input_values).logits

        # take the probs
        probs = torch.softmax(logits, dim=-1)
        top2_probs, top2_ids = torch.topk(probs, k=2, dim=-1)

        # take argmax and decode, greedy decoding
        predicted_ids = torch.argmax(logits, dim=-1)

        # Manually collapse repeated tokens for each sequence in the batch
        collapsed_ids = []
        for seq in predicted_ids:
            seq = seq.tolist()
            collapsed_seq = self.collapse_repeats(seq)
            collapsed_ids.append(collapsed_seq)

        # Decode the collapsed token sequences to get phoneme transcription strings
        # (The tokenizer's decode method will convert token ids to phoneme symbols)
        transcription = [self.processor.tokenizer.decode(ids) for ids in collapsed_ids]
        transcription = self.model_output_processing(transcription) # convert and filter our output


        return transcription
