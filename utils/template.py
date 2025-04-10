from dataclasses import dataclass


@dataclass
class Template:
    system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    assistant_format = '{content}<|eot_id|>'
    system = "You are a knowledge-based assistant. Use the following knowledge context to answer questions or engage in conversation.\nKnowledge:"
    stop_word = '<|eot_id|>'


K2T_PROMPTS = [
    "Describe the knowledge context.",
    "Provide a detailed description of the knowledge context.",
    "Can you explain what the knowledge context consisted of?",
    "Thoroughly outline the details of the knowledge context used.",
    "Provide a comprehensive overview of the knowledge used in the context.",
    "Elaborate on the content of the knowledge context used.",
    "What information did the knowledge context contain? Please describe in detail.",
    "Provide an in-depth explanation of the content covered in the knowledge context.",
]

T2K_PROMPTS = [
    "Identify and list the key information present in the detailed text.",
    "Extract the core pieces of key information that summarize the knowledge provided.",
    "What are the main themes or key pieces of information depicted in the text? List them.",
    "Summarize the text into essential pieces of key information.",
    "Distill the primary pieces of information from the text into concise descriptors.",
    "From the detailed knowledge described, what are the central pieces of key information? Please enumerate.",
    "Determine the main pieces of key information that capture the essence of the text provided.",
    "What key pieces of information would you use to index the information described here?",
]