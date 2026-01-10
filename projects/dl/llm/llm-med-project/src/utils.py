


    
def build_text(example):
    prompt_text = example["input"]
    answer_text = example["output"]
    full_text = prompt_text + answer_text
    return {"text":full_text}