from transformers import BertConfig, BertModel, AutoTokenizer
import torch


if __name__ == '__main__':
    config = BertConfig()  # Building the config
    model = BertModel(config)  # Building the model from the config, it initializes the model from random values
    print(config)

    model = BertModel.from_pretrained("bert-base-cased")  # Pre-trained model
    model.save_pretrained("directory_on_my_computer")  # This saves: config.json (architecture, attributes, metadata) and pytorch_model.bin (weights)

    sequences = ["Hello!", "Cool.", "Nice!"]
    encoded_sequences = [
        [101, 7592, 999, 102],
        [101, 4658, 1012, 102],
        [101, 3835, 999, 102],
    ]

    model_inputs = torch.tensor(encoded_sequences)
    output = model(model_inputs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    print(tokenizer("Using a Transformer network is simple"))
    tokenizer.save_pretrained("directory_on_my_computer")

    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)  # Tokens before being converted to integers
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
    print(decoded_string)

