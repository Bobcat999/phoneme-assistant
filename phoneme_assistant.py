from openai import OpenAI
from dotenv import load_dotenv
from phoneme_extractor import PhonemeExtractor
from word_extractor import WordExtractor
from text_to_audio import ElevenLabsAPIClient
import os
import pandas as pd

class PhonemeAssistant:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reset_conversation_history()
        self.phoneme_extractor = PhonemeExtractor()
        self.word_extractor = WordExtractor()
        self.tts = ElevenLabsAPIClient()
    
    def reset_conversation_history(self):
        self.conversation_history = [ # Reset conversation history to just be the initial system prompt
            {
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": "You are a world-class personalized phonics assistant designed to help elementary school children improve their pronunciation through targeted phonics practice.\n\nYou will receive an input object in the following format:\n\nInput Format:\n{\n  \"attempted_scentence\":\"<attempted_scentence>\",\n  \"pronunciation\": [\n    {\n      \"type\": \"match\",\n      \"predicted_word\": \"<predicted_word>\",\n      \"ground_truth_word\": \"<ground_truth_word>\",\n      \"phonemes\": [\"<phoneme1>\", \"<phoneme2>\", ...],\n      \"ground_truth_phonemes\": [\"<ground_truth_phoneme1>\", \"<ground_truth_phoneme2>\", ...],\n      \"per\": <percentage_error>,\n      \"missed\": [],\n      \"added\": [],\n      \"substituted\": []\n    },\n    {\n      \"type\": \"deletion\",\n      \"predicted_word\": null,\n      \"ground_truth_word\": \"<ground_truth_word>\",\n      \"phonemes\": null,\n      \"ground_truth_phonemes\": null,\n      \"per\": null,\n      \"missed\": null,\n      \"added\": null,\n      \"substituted\": null,\n      \"error\": \"Word missing in prediction.\"\n    },\n    {\n      \"type\": \"substitution\",\n      \"predicted_word\": \"<predicted_word>\",\n      \"ground_truth_word\": \"<ground_truth_word>\",\n      \"phonemes\": [\"<phoneme1>\", \"<phoneme2>\", ...],\n      \"ground_truth_phonemes\": [\"<ground_truth_phoneme1>\", \"<ground_truth_phoneme2>\", ...],\n      \"per\": <percentage_error>,\n      \"missed\": [],\n      \"added\": [],\n      \"substituted\": [(\"<incorrect_phoneme>\", \"<correct_phoneme>\"), ...]\n    }\n  ],\n  \"highest_per_word\": {\n    \"type\": \"<type>\",\n    \"predicted_word\": \"<predicted_word>\",\n    \"ground_truth_word\": \"<ground_truth_word>\",\n    \"phonemes\": [\"<phoneme1>\", \"<phoneme2>\", ...],\n    \"ground_truth_phonemes\": [\"<ground_truth_phoneme1>\", \"<ground_truth_phoneme2>\", ...],\n    \"per\": <percentage_error>,\n    \"missed\": [],\n    \"added\": [],\n    \"substituted\": [(\"<incorrect_phoneme>\", \"<correct_phoneme>\")],\n    \"error\": \"<optional_error>\"\n  }\n}\nTask:\nTargeted Feedback:\n\nFocus on the highest error rate (provided by the 'highest_per_word' section of your input):\n\nIf the word with the highest error is a substitution, compare the student's substitution to the correct phoneme and explain in child-friendly terms why it is incorrect.\n\nIf the student omitted a word (indicated by a \"deletion\"), provide feedback on the missing word and explain how it affects the sentence.\n\nExplanation: Use simple, relatable language to describe the mistake, and give examples of words that share the correct pronunciation.\n\nContextualize the Mistake: Use context to explain how the correct phoneme is used in real words, ensuring that the student can hear and practice the difference.\n\nCompliment: Allways give the student a compliment before and after your constructive feedback, forming a compliment sandwich.\n\nNew Practice Sentence:\n\nIf the student made a mistake in pronunciation:\n\nCreate a new sentence that reinforces the corrected sound by including words that naturally use the corrected phoneme.\n\nKeep the sentence short (5-8 words) and engaging, helping the student practice the targeted sound.\n\nIf the student's pronunciation was strong:\n\nIntroduce a new sentence that targets a different phoneme to continue challenging the student. Make sure the sentence is still decodable and meaningful.\n\nExample Output (if a substitution error occurred):\n{\n  \"feedback\": \"Nice try! You said 'doge,' but it should sound like 'dog.' You made the 'g' sound too soft, like in 'judge.' Try saying 'dog' with a hard 'g'—like in 'go' or 'get.' Keep practicing and you'll get it!\",\n  \"sentence\": \"The dog is big and brown.\"\n}\nExample Output (if a deletion error occurred):\n{\n  \"feedback\": \"Oops, you missed the word 'the' in your sentence! It's an important word that helps make the sentence clear. Try saying, 'The quick brown fox jumped over the lazy dog.'\",\n  \"sentence\": \"The quick brown fox jumped over the lazy dog.\"\n}\nExample Output (if no significant mistake occurred):\n{\n  \"feedback\": \"Great job! Your pronunciation is getting clearer. Now, let’s try a new sound. Can you say this sentence and focus on making the 'oo' in 'moon' sound just right?\",\n  \"sentence\": \"The moon shines bright at night.\"\n}"
                    }
                ]
            }
        ]
    
    def getPhonemes(self, text):
        phonemes = self.client.text_to_phonemes.convert(
            text=text,
            voice_id="nPczCjzI2devNBz1zQrb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",)
        return phonemes

    def get_response(self, attempted_sentence:str, results:list, highest_per_word: pd.Series) -> dict:
        # format the user input for our model
        user_input = {
            "attempted": attempted_sentence,
            "pronunciation": results,
            "highest_per_word": highest_per_word.to_dict()
        }

        # add the user input to the conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # get the response from the model
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history,
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response_schema",
                "schema": {
                "type": "object",
                "required": [
                    "feedback",
                    "sentence"
                ],
                "properties": {
                    "feedback": {
                    "type": "string",
                    "description": "Specific feedback provided to the student to help them improve their reading pronunciation."
                    },
                    "sentence": {
                    "type": "string",
                    "description": "A decodable sentence based on the mistakes made in the student's pronunciation."
                    }
                },
                "additionalProperties": False
                },
                "strict": True
            }
            },
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        model_response = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": model_response})
        
        return model_response

    def record_audio_and_get_response(self, attempted_sentence, verbose=False):
        results, ground_truth_phonemes = record_and_process_pronunciation(attempted_sentence, phoneme_extraction_model=self.phoneme_extractor, word_extraction_model=self.word_extractor)
        df, highest_per_word = analyze_results(results)
        if verbose:
            print("Dataframe: ", df)
            print("Highest error word: ", highest_per_word)
        model_response = self.get_response(attempted_sentence=attempted_scentance, results=results, highest_per_word=highest_per_word)
        return model_response
    
    def feedback_to_audio(self, feedback: str):
        return self.tts.getAudio(feedback, playAudio=True)




# Default running behavior
if __name__ == "__main__":
    from audio_recording import record_and_process_pronunciation
    from process_audio import analyze_results
    from phonecodes.phonecodes import ipa2arpabet
    import json

    # AI models
    phoneme_assistant = PhonemeAssistant()

    print("Welcome to the Phonics Assistant! Let's get started.")

    attempted_sentence = input("Enter sentence to start with: ").lower()

    # user loop
    while True:
        output = phoneme_assistant.record_audio_and_get_response(attempted_sentence, verbose=True)

        output_json = json.loads(output)
        print("\nFeedback: ")
        print(output_json['feedback'])

        print("\n New sentence to say: ")
        print(output_json['sentence'])

        attempted_scentance = output_json['sentence'].replace(".", "").lower()

        audio_feedback = f'{output_json["feedback"]}'
        phoneme_assistant.feedback_to_audio(audio_feedback)

        if input("Do you want to continue? (yes/no): ") != "yes":
            break