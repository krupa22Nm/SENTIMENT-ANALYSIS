from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import torch

def fine_tune_distilbert(df):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=3,
        id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
        label2id={'negative': 0, 'neutral': 1, 'positive': 2}
    )
    
    # Tokenize
    def tokenize(batch):
        return tokenizer(batch['clean_text'], truncation=True, padding=True, max_length=128)
    
    dataset = Dataset.from_pandas(df[['clean_text', 'sentiment']])
    dataset = dataset.map(lambda x: {'labels': label2id[x['sentiment']]})
    dataset = dataset.map(tokenize, batched=True)
    
    # Train
    training_args = TrainingArguments(
        output_dir='./models/distilbert_finetuned',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_test_split(test_size=0.2)['train'],
        eval_dataset=dataset.train_test_split(test_size=0.2)['test'],
    )
    
    trainer.train()
    trainer.save_model()
    return model, tokenizer