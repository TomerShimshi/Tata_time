import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments,TextDataset,DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# Path to your local text file
file_path = 'rebe.txt'

# Load your custom text dataset
#dataset = load_dataset('text', data_files={'train': [file_path]}, sample_by="document")#, split='train')

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenized_datasets =TextDataset(tokenizer=tokenizer, file_path=file_path,block_size=32)
data_collater = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
# Choose an existing token as the padding token
tokenizer.pad_token = tokenizer.eos_token  # You can choose a different token if needed
# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], padding=True, truncation=True)
    #inputs['train'] = inputs['text']
    return inputs
    
    
     

#tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./fine-tuned-model',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)
# Custom training step to calculate loss
def compute_loss(model, inputs):
    labels = inputs["input_ids"]
    outputs = model(**inputs, labels=labels)
    return outputs.loss
# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets.examples[:int(len(tokenized_datasets)*0.8)],
    eval_dataset= tokenized_datasets.examples[int(len(tokenized_datasets)*0.8):],
    data_collator=data_collater
)
import os
os.environ['WANDB_DISABLED'] = "true"
print("starting training")

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')