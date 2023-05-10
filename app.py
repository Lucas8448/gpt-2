import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
input_text = "Hello!"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones_like(input_ids)
pad_token_id = tokenizer.pad_token_id
mask = torch.tensor(input_ids == pad_token_id, dtype=torch.bool)
attention_mask = attention_mask.masked_fill(mask, 0)
output = model.generate(input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1000)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
