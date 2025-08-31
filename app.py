import gradio as gr
import os

def predict(query):
    return f"Mock SQL: {query}"

iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
