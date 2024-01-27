from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from transformers import TrainingArguments, Trainer,TextDataset,DataCollatorForLanguageModeling
from datasets import load_dataset
import evaluate
from transformers import LlamaTokenizer, LlamaForCausalLM,BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
    



file_path = 'Rebe_Q_and_A_dataset_just_rebe_questions.txt'
metric = evaluate.load("accuracy")

with open(file_path, "r", encoding='utf-8' ) as f:
    dataset = f.read()

model_path = 'openlm-research/open_llama_3b'


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,#64,
    bias="none",
    task_type="CAUSAL_LM",
)


bnb_config = BitsAndBytesConfig(load_in_8bit = True)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
model = LlamaForCausalLM.from_pretrained(
    model_path, quantization_config=bnb_config, device_map="auto",
)
tokenized_datasets =TextDataset(tokenizer=tokenizer, file_path=file_path,block_size=34)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 0 if torch.cuda.is_available()==False else torch.cuda.device_count()
data_collater = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
# Choose an existing token as the padding token
tokenizer.pad_token = tokenizer.eos_token  # You can choose a different token if needed
# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], padding=True, truncation=True)
    #inputs['train'] = inputs['text']
    return inputs

model.add_adapter(peft_config)
  
training_args = TrainingArguments(output_dir="fine-tuned-model_v2",
evaluation_strategy="epoch",learning_rate=2e-4,
 per_device_train_batch_size=34,#1,
 gradient_accumulation_steps=34,#4,
overwrite_output_dir=True,
 per_device_eval_batch_size=4,
 num_train_epochs=10,
 report_to= "wandb",
 warmup_steps= len(tokenized_datasets)//6,
 remove_unused_columns= False,
 #load_best_model_at_end=True,
 fp16=True,
 logging_steps=10,
    eval_steps=10,
 save_strategy='epoch',
 gradient_checkpointing=True,
 save_total_limit=2,
  )


trainer = Trainer(
    model=model,
    args=training_args,
     train_dataset=tokenized_datasets.examples[:int(len(tokenized_datasets)*0.85)],
    eval_dataset= tokenized_datasets.examples[int(len(tokenized_datasets)*0.85):],
    data_collator= data_collater,  
)

#trainer = SFTTrainer(
#    model=model,
#    train_dataset=dataset[:int(0.85 * len(dataset))],#tokenized_datasets.examples[:int(len(tokenized_datasets)*0.85)],
#    eval_dataset= dataset[int(0.85 * len(dataset)):] ,#tokenized_datasets.examples[int(len(tokenized_datasets)*0.85):],
#    peft_config=peft_config,
#    dataset_text_field="text",
#    data_collator= data_collater,
#    args=training_args
#)

import os
os.environ['WANDB_DISABLED'] = "true"
print("starting training")

trainer.train()

model.save_pretrained('./best_fine-tuned-model_with_QA_llama')
tokenizer.save_pretrained('./best_fine-tuned-model_with_QA_llama')
print(f"the best state is {trainer.state.best_model_checkpoint}")