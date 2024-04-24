import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == '__main__':
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([ids])  # Insert into a list to batch
    model(input_ids)

    # if the sequences are different in shape, padding needs to be done
    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]

    print(model(torch.tensor(sequence1_ids)).logits)
    print(model(torch.tensor(sequence2_ids)).logits)
    print(model(torch.tensor(batched_ids)).logits)  # The 2nd output is different because of the padding

    # Attention masks
    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]
    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits)

    # Long sequences: use a model that supports the length or truncate it
    # sequence = sequence[:max_sequence_length]

    # Using the Tokenizer API
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
    model_inputs = tokenizer(sequences)

    # Types of padding
    model_inputs = tokenizer(sequences, padding="longest")  # Will pad the sequences up to the maximum sequence length
    model_inputs = tokenizer(sequences, padding="max_length")  # Will pad the sequences up to the model max length (512 for BERT or DistilBERT)
    model_inputs = tokenizer(sequences, padding="max_length", max_length=8)   # Will pad the sequences up to the specified max length

    # Truncate
    model_inputs = tokenizer(sequences, truncation=True)  # Will truncate the sequences that are longer than the model max length
    model_inputs = tokenizer(sequences, max_length=8, truncation=True)

    # Frameworks: can be pt, tf, np
    model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
    model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
    model_inputs = tokenizer(sequences, padding=True, return_tensors="np")


