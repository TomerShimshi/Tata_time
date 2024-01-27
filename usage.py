from transformers import AutoTokenizer, AutoModelForCausalLM
  
tokenizer = AutoTokenizer.from_pretrained("Norod78/hebrew-gpt_neo-xl")
model = AutoModelForCausalLM.from_pretrained("Norod78/hebrew-gpt_neo-xl", pad_token_id=tokenizer.eos_token_id)

prompt_text = "אני אוהב שוקולד ועוגות"
max_len = 100
sample_output_num = 3
seed = 1000

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 0 if torch.cuda.is_available()==False else torch.cuda.device_count()

print(f"device: {device}, n_gpu: {n_gpu}")

np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

model.to(device)

encoded_prompt = tokenizer.encode(
    prompt_text, add_special_tokens=False, return_tensors="pt")

encoded_prompt = encoded_prompt.to(device)

if encoded_prompt.size()[-1] == 0:
        input_ids = None
else:
        input_ids = encoded_prompt

print("input_ids = " + str(input_ids))

if input_ids != None:
  max_len += len(encoded_prompt[0])
  if max_len > 2048:
    max_len = 2048

print("Updated max_len = " + str(max_len))

stop_token = "<|endoftext|>"
new_lines = "\
\
\
"

sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=max_len, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=sample_output_num
)

print(100 * '-' + "\
\t\tOutput\
" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):

  text = tokenizer.decode(sample_output, skip_special_tokens=True)
  
  # Remove all text after the stop token
  text = text[: text.find(stop_token) if stop_token else None]

  # Remove all text after 3 newlines
  text = text[: text.find(new_lines) if new_lines else None]

  print("\
{}: {}".format(i, text[::-1]))
  print("\
" + 100 * '-')