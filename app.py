import os
os.system("pip install transformers datasets sentencepiece gradio --quiet")

# ✅ Imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import gradio as gr
import torch
import re

# ✅ Load schema-aware model
model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Load Spider validation set
dataset = load_dataset("spider")
dev_ds = dataset["validation"]

# ✅ Helpers for prompt creation and SQL generation
def make_prompt(question, schema):
    return f"Question: {question} Schema: {schema}"

# ✅ Post-processing to fix common SQL issues
def clean_sql(sql):
    # Fix 1: Remove redundant GROUP BY if WHERE filters the group
    sql = re.sub(
        r"WHERE\s+(\w+)\s*=\s*['\"]?([\w\s]+)['\"]?\s+GROUP BY\s+\1",
        r"WHERE \1 = '\2'",
        sql,
        flags=re.IGNORECASE
    )

    # Fix 2: Ensure grouped column is included in SELECT clause
    match = re.search(r"GROUP BY\s+(\w+)", sql, flags=re.IGNORECASE)
    if match:
        group_col = match.group(1)
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, flags=re.IGNORECASE)
        if select_match:
            select_cols = select_match.group(1)
            # Normalize columns for comparison
            select_cols_lower = [col.strip().lower() for col in select_cols.split(",")]
            if group_col.lower() not in select_cols_lower:
                new_select = f"{group_col}, {select_cols}"
                sql = re.sub(r"SELECT\s+(.*?)\s+FROM", f"SELECT {new_select} FROM", sql, flags=re.IGNORECASE)

    # Fix 3: Correct aggregation functions (e.g., "max salary" → "MAX(salary)")
    agg_funcs = ["max", "min", "avg", "sum", "count"]
    for func in agg_funcs:
        pattern = rf"\b{func}\s+(\w+)\b"
        sql = re.sub(pattern, lambda m: f"{func.upper()}({m.group(1)})", sql, flags=re.IGNORECASE)

    return sql

def generate_sql(question, schema):
    prompt = make_prompt(question, schema)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_sql(sql)

# ✅ Evaluation function
def evaluate_spider(dev_ds, n_eval=100):
    hits = 0
    for i in range(min(n_eval, len(dev_ds))):
        item = dev_ds[i]
        question = item["question"]
        schema = item.get("db_schema", "") or item.get("schema", "")
        gold_sql = item["query"]
        pred_sql = generate_sql(question, schema)
        if pred_sql.strip().lower() == gold_sql.strip().lower():
            hits += 1
    acc = hits / n_eval
    print(f"✅ Exact Match Accuracy: {hits}/{n_eval} = {acc*100:.2f}%")
    return acc

# ✅ Run evaluation on Spider dev set
evaluate_spider(dev_ds, n_eval=100)

# ✅ Gradio Interface
def gradio_sql(question, schema):
    return generate_sql(question, schema)

gr.Interface(
    fn=gradio_sql,
    inputs=[
        gr.Textbox(label="Natural Language Question", lines=2, placeholder="e.g. List all customers in California"),
        gr.Textbox(label="Database Schema", lines=6, placeholder="e.g. CREATE TABLE customers (id INT, state TEXT, ...)"),
    ],
    outputs=gr.Textbox(label="Generated SQL Query"),
    title="🧠 Text-to-SQL Generator (Spider-trained T5)",
    description="Enter a question and schema to generate SQL. Powered by T5 fine-tuned on Spider.",
).launch(share=True)

