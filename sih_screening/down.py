from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Choose model (T5 or FLAN-T5)
model_name = "google/flan-t5-base"  # or "google/flan-t5-base"

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save locally for offline use
save_path = "./models/reason_offline"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Model and tokenizer saved to {save_path}")
