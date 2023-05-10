import os
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the recursion depth limit
depth_limit = 2

# Fetch text and links from Wikipedia pages recursively


def fetch_wikipedia_text(url, depth=0):
    if depth >= depth_limit:
        return ""

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text from paragraphs
    paragraphs = soup.find_all("p")
    text_data = " ".join([p.text for p in paragraphs])

    # Extract links from paragraphs
    links = [a["href"] for p in paragraphs for a in p.find_all("a", href=True)]
    print("page links", links)
    full_links = ["https://en.wikipedia.org" +
                  link for link in links if link.startswith("/wiki/")]

    # Recursively fetch text from linked pages
    for link in full_links:
        text_data += " " + fetch_wikipedia_text(link, depth=depth+1)

    return text_data



# Fetch text from Wikipedia's featured article of the day
url = "https://en.wikipedia.org/wiki/Main_Page"
text_data = fetch_wikipedia_text(url)

# Save the training data to a file
os.makedirs("training_data", exist_ok=True)
train_file = "training_data/training_data.txt"
with open(train_file, "w", encoding="utf-8") as f:
    f.write(text_data)

# Prepare the training dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)

# Prepare the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Load the GPT-2 model configuration
config = GPT2Config.from_pretrained("gpt2")

# Create a new GPT-2 model
model = GPT2LMHeadModel(config)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()
