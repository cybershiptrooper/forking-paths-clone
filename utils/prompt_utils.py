import re
from transformers import PreTrainedTokenizer
from utils.utils import MODEL_METADATA

TODAY_DATE_STRING = "Today Date: 12 December 2024\n"

MC_INSTRUCTION_PROMPT = """Answer the following multiple choice question. **Include the letter in your final answer.**
{question}"""

def fix_date(prompt : str):
    return re.sub("Today Date:.*\n", TODAY_DATE_STRING, prompt)

def get_prompt(tokenizer, question, answer_prefix=None, multiple_choice=False):
    if multiple_choice:
        prompt = MC_INSTRUCTION_PROMPT.format(question=question)
    else:
        prompt = question
    chat = [
        {'role': 'user', 'content': prompt}
    ]
    
    if answer_prefix is None:
        add_generation_prompt = True
    else:
        chat += [
            {'role': 'assistant', 'content': answer_prefix}
        ]
        add_generation_prompt = False
    
    prompt = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        continue_final_message=not add_generation_prompt,
        add_generation_prompt=add_generation_prompt
    )
    return fix_date(prompt)

def get_cot_prompt(tokenizer : PreTrainedTokenizer, question):
    if MODEL_METADATA[tokenizer.name_or_path]['reasoning']:
        answer_prefix = None
    else:
        answer_prefix = "Let's think step by step."

    return get_prompt(tokenizer, question, answer_prefix=answer_prefix, multiple_choice=True)

def get_alignment_prompt(tokenizer, question, alignment_type=None):
    if alignment_type is None:
        answer_prefix = None
    elif alignment_type == "misaligned":
        answer_prefix = "Sure,"
    elif alignment_type == "aligned":
        answer_prefix = "Sorry,"
    return get_prompt(tokenizer, question, answer_prefix=answer_prefix, multiple_choice=False)