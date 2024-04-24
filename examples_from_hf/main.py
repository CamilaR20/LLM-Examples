from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
import evaluate
import time


def tokenize_function(example):
    # To avoid loading the full dataset in RAM
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


if __name__ == '__main__':
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    # Train model with a few examples
    # sequences = ["I've been waiting for a HuggingFace course my whole life.", "This course is amazing!"]
    # batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    # batch["labels"] = torch.tensor([1, 1])
    # optimizer = AdamW(model.parameters())
    # loss = model(**batch).loss
    # loss.backward()
    # optimizer.step()

    # Train model with MRPC dataset
    raw_datasets = load_dataset("glue", "mrpc")  # Contains train, validation and test dataset
    raw_train_dataset = raw_datasets["train"]
    # print(raw_train_dataset[0])
    # print(raw_train_dataset.features)  # Type of each column

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # This adds new fields to the dataset
    # print(tokenized_datasets)

    # Dynamic Padding: the collate function puts samples inside a batch, it pads each sample in batch to the longest
    # length within it to speed up training and reduce extra padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments("test-trainer")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # trainer.train()
    #
    # predictions = trainer.predict(tokenized_datasets["validation"])  # predictions (logits), label_ids and metrics
    # print(predictions.predictions.shape, predictions.label_ids.shape)
    # preds = np.argmax(predictions.predictions, axis=-1)  # Get label from logits
    #
    # metric = evaluate.load("glue", "mrpc")
    # print(metric.compute(predictions=preds, references=predictions.label_ids))
