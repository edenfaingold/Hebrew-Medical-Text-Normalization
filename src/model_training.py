import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, Trainer, TrainingArguments
from data_preprocessing import preprocess_data, create_label_mappings, encode_labels


os.environ['TRANSFORMERS_OFFLINE'] = '1'
tokenizer_dir = './dictalm2.0'
model_name = "dicta-il/dictalm2.0"
data_directory = "data"


def tokenize_sentences(sentences, tokenizer):
    """ Tokenize sentences using the provided tokenizer """
    return tokenizer(
        sentences,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )


class SentenceDataset(torch.utils.data.Dataset):
    """ Dataset class for tokenized sentences """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def load_tokenizer():
    """ Load tokenizer from the tokenizer directory """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_dataset(sentences, labels, tokenizer):
    """ Create a dataset from tokenized sentences and labels """
    encodings = tokenize_sentences(sentences, tokenizer)
    return SentenceDataset(encodings, labels)


def initialize_model(num_labels, id_to_label, label_to_id, tokenizer):
    """ Initialize a model for token classification """
    config = AutoConfig.from_pretrained(
        tokenizer_dir,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )
    model = AutoModelForTokenClassification.from_pretrained(
        tokenizer_dir,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_training_args():
    """ Get training arguments for the model """
    return TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True
    )


def train_model(model, training_args, dataset, tokenizer):
    """ Train the model on the dataset """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    return trainer


def save_model(model, tokenizer, directory):
    """ Save the model and tokenizer to the specified directory """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    print(f"Model saved to {directory}")


def main():
    files, sentences, labels = preprocess_data(data_directory)
    label_list, label_to_id, id_to_label = create_label_mappings(labels)
    num_labels = len(label_list)
    labels_encoded = encode_labels(labels, label_to_id)
    tokenizer = load_tokenizer()
    dataset = create_dataset(sentences, labels_encoded, tokenizer)
    model = initialize_model(num_labels, id_to_label, label_to_id, tokenizer)
    training_args = get_training_args()
    trainer = train_model(model, training_args, dataset, tokenizer)
    trainer.train()
    save_model(model, tokenizer, './fine_tuned_model')


if __name__ == "__main__":
    main()
