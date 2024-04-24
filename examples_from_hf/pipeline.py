from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

if __name__ == '__main__':
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")  # If no type of tensor is specified a list of lists will be returned
    print(inputs)

    model = AutoModel.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    # Model with atttention head for the sequence classification task
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.logits.shape)  # 2 sentences and 2 labels
    print(outputs.logits)  # Raw, unnormalized scores
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    print(model.config.id2label)  # Labels from model