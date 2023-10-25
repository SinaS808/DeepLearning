import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Laden des Modells und des Tokenizers
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

st.title("Chatbot mit Huggingface und Streamlit")

# Input und Output Textareas
input_text = st.text_area("Du:", "")
output_text = ""

if st.button("Antworten"):
    # Codieren der Eingabe und Generieren einer Antwort
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

st.text_area("Chatbot:", value=output_text, disabled=True)