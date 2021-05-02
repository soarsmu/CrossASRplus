import numpy as np
from crossasr.constant import NUM_LABELS, FAILED_TEST_CASE

from crossasr.estimator import Estimator
from crossasr.text import Text

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import Trainer, TrainingArguments
from scipy.special import softmax


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class HuggingFaceTransformer(Estimator):
    def __init__(self, name, max_sequence_length=128):
        Estimator.__init__(self, name=name)

        ## init boiler plate

        ## init model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=NUM_LABELS)

        ## init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.max_sequence_length = max_sequence_length

    def fit(self, X: [str], y: [int]):

        self.model.to(self.device)
        self.model.train()

        train_texts = X
        train_labels = y

        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=self.max_sequence_length)
        train_dataset = HuggingFaceDataset(train_encodings, train_labels)

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=1,              # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            learning_rate=5e-05,
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )

        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=self.model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset         # training dataset
        )

        trainer.train()

    def predict(self, X: [str]):
        self.model.eval()

        test_texts = X
        test_labels = [0] * len(X)

        test_encodings = self.tokenizer(
            test_texts, truncation=True, padding=True, max_length=self.max_sequence_length)
        test_dataset = HuggingFaceDataset(test_encodings, test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False)

        res = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels)
            preds = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
            res.extend(preds)
        res = np.array(res)

        failed_probability = res[:, FAILED_TEST_CASE]

        # print(failed_probability[:10])

        del test_loader
        torch.cuda.empty_cache()

        return failed_probability

