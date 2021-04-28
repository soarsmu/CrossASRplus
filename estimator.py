
# class Classifier:
# 	def __init__(name):
# 	def classify(List[:text])

# # exract text and audio feature into emebedding feature consumable by Estimator


# class FeatureExtractor:

# 	# estimator to predict how likely the input to be a failed test case

import numpy as np
from constant import NUM_LABELS, FAILED_TEST_CASE

from text import Text

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from scipy.special import softmax


class Estimator:
    def __init__(self, name:str):
        self.name = name
    
    def fit(self, X:[str], y:[int]):
        raise NotImplementedError()

    def predict(self, X:[str]):
        raise NotImplementedError()


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class HuggingFaceTransformer(Estimator):
    def __init__(self, name):
        Estimator.__init__(self, name=name)

        ## init boiler plate

        ## init model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=NUM_LABELS)
        
        ## init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def fit(self, X:[str], y:[int]) :
        
        self.model.to(self.device)
        self.model.train()

        train_texts = X
        train_labels = y

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        train_dataset = HuggingFaceDataset(train_encodings, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

        optim = AdamW(self.model.parameters(), lr=5e-5)

        for _ in range(1):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
        del train_loader
        torch.cuda.empty_cache()

    def predict(self, X: [str]):
        self.model.eval()

        test_texts = X
        test_labels = [0] * len(X)

        test_encodings = self.tokenizer(
            test_texts, truncation=True, padding=True)
        test_dataset = HuggingFaceDataset(test_encodings, test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, shuffle=True)

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

        failed_probability = res[:,FAILED_TEST_CASE]

        # print(failed_probability[:10])
        
        del test_loader
        torch.cuda.empty_cache()

        return failed_probability

def create_huggingface_estimator_by_name(name:str):
    # https://huggingface.co/transformers/custom_datasets.html
    return HuggingFaceTransformer(name=name)


