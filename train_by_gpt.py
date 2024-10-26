from datasets import load_dataset
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from transformers import GPT2Model, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

# with lots of human debugging

class GPT2Classifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        token_0 = self.tokenizer.encode("0", add_special_tokens=False)[0]
        token_1 = self.tokenizer.encode("1", add_special_tokens=False)[0]
        
        orig_embed = self.gpt2.wte.weight
        
        self.classifier = nn.Linear(self.gpt2.config.n_embd, 2)
        with torch.no_grad():
            self.classifier.weight.data[0] = orig_embed[token_0]
            self.classifier.weight.data[1] = orig_embed[token_1]
    
    def forward(self, input_ids):
        outputs = self.gpt2(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled = last_hidden_state[:, -1, :]
        return self.classifier(pooled)

class SciQDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.texts = [
            f"Question: {item['question']} Answer: {item['answer']}" 
            for item in data
        ]
        self.labels = torch.tensor([item['label'] for item in data])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'label': label
        }
from torch.nn.utils.rnn import pad_sequence
import itertools

def train_model(model, train_ds, num_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Reduced learning rate
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    epochs = 2
    nsteps = len(train_ds) * epochs // batch_size
    step = 0
    model.train()
    it = itertools.chain.from_iterable(itertools.repeat(train_ds, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}
    while step < nsteps:
        loss_tot = 0
        all_logits = []
        all_labels = []
        for i in range(batch_size):
            try:
                mbatch = next(it)
            except StopIteration:
                break
            input_ids = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch])
                .transpose(
                    0,
                    1,
                )
                .to(device)
            )
            labels = torch.tensor([ex["label"] for ex in mbatch]).to(device)

            logits = model(input_ids)

            all_logits.extend(logits.to(device))
            all_labels.extend(labels)
        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        loss = criterion(all_logits, all_labels)
        loss_tot += loss.item()
        loss.backward()
        losses.append(loss_tot)
        accuracies.append(
            torch.mean(
                (torch.argmax(all_logits, dim=1) == torch.tensor(all_labels)).to(
                    torch.float32
                )
            ).item()
        )
        print({
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "train_accuracy": accuracies[-1],
            })
        optimizer.step()
        optimizer.zero_grad()
        step +=1


def generate_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    return predictions

def convert_to_binary_format(dataset) -> List[Dict]:
    binary_data = []
    for item in dataset:
        question = item['question']
        correct_answer = item['correct_answer']
        incorrect_answers = [
            item['distractor1'],
            item['distractor2'],
            item['distractor3']
        ]
        
        if random.random() < 0.5:
            binary_data.append({
                'question': question,
                'answer': correct_answer,
                'label': 1,
                'question_id': hash(question)
            })
        else:
            selected_incorrect = random.choice(incorrect_answers)
            binary_data.append({
                'question': question,
                'answer': selected_incorrect,
                'label': 0,
                'question_id': hash(question)
            })
    
    return binary_data

def main():
    device = torch.device('mps')
    print(f"Using device: {device}")
    
    print("Loading SciQ dataset...")
    dataset = load_dataset("sciq", split="train")
    binary_data = convert_to_binary_format(dataset)
    # print(f"{binary_data=}")
    
    # First split: 95% train, 5% eval
    train_data, eval_data = train_test_split(binary_data, test_size=0.1, random_state=42)
    
    # Second split: Split train data in half for weak supervision
    weak_train_data, strong_train_data = train_test_split(train_data, test_size=0.5, random_state=42)
    
    print("\nDataset splits:")
    print(f"Weak train size: {len(weak_train_data)}")
    print(f"Strong train size: {len(strong_train_data)}")
    print(f"Eval size: {len(eval_data)}")
    
    # Train weak model (GPT2)
    print("\nTraining weak model (GPT2)...")
    weak_model = GPT2Classifier('gpt2').to(device)
    weak_train_dataset = SciQDataset(weak_train_data, weak_model.tokenizer)
    # weak_train_loader = DataLoader(weak_train_dataset, batch_size=16, shuffle=True)
    
    train_model(weak_model, weak_train_dataset, num_epochs=2, device=device)
    
    # Generate weak labels for strong training data
    print("\nGenerating weak labels...")
    strong_dataset = SciQDataset(strong_train_data, weak_model.tokenizer)
    strong_loader = DataLoader(strong_dataset, batch_size=32)
    weak_labels = generate_predictions(weak_model, strong_loader, device)
    
    # Update labels in strong training data
    for item, weak_label in zip(strong_train_data, weak_labels):
        item['label'] = int(weak_label)
    
    print("\nWeak label distribution:")
    unique_labels, counts = np.unique(weak_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} ({count/len(weak_labels):.2%})")
    
    # Train strong model (GPT2-medium)
    print("\nTraining strong model (GPT2-medium)...")
    strong_model = GPT2Classifier('gpt2-medium').to(device)
    strong_train_dataset = SciQDataset(strong_train_data, strong_model.tokenizer)
    # strong_train_loader = DataLoader(strong_train_dataset, batch_size=16, shuffle=True)
    
    train_model(strong_model, strong_train_dataset, num_epochs=2, device=device)
    
    # Evaluate on eval set
    print("\nEvaluating strong model...")
    eval_dataset = SciQDataset(eval_data, strong_model.tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=32)
    
    strong_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = strong_model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"\nEvaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()