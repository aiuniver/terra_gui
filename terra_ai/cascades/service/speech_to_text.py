from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch


def wav2vec2_large_russian(**nothing):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

    def fun(speech_array):
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences = processor.batch_decode(predicted_ids)

        return predicted_sentences[0].lower()

    return fun
