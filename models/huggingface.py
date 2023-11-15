from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, LlamaTokenizer
from anchor import checkpoints_root


def build_model_signature(model_type, model_size, instruct=''):
    if model_type == "opt":
        # ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b"]
        return f"facebook/opt-{model_size}"
    if model_type == "gpt2":
        # ["sm", "medium", "large", "xl"]
        if model_size == "sm":
            return "gpt2"
        return f"gpt2-{model_size}"
    if model_type == "e-gpt":
        # ["neo-125M", "neo-1.3B", "neo-2.7B", "j-6B", "neox-20b"]
        return f"EleutherAI/gpt-{model_size}"
    if model_type == "bloom":
        # ["560m", "1b1", "1b7", "3b", "7b1"]
        return f"bigscience/bloom-{model_size}"
    if model_type == 'falcon':
        return f"tiiuae/falcon-{model_size}{instruct}"
    if model_type == 'llama':
        return f"yahma/llama-7b-hf"
    if model_type == 'llama-33b':
        return f"alexl83/LLaMA-33B-HF"
    if model_type == 'vicuna':
        return f"lmsys/vicuna-{model_size}{instruct}"
    if model_type == 'llama-2':
        return f"meta-llama/Llama-2-7b-chat-hf"
        
def build_tokenizer(model_type, model_size, padding_side="left", use_fast=False):
    sign = build_model_signature(model_type, model_size)

    if 'llama' in model_type:
        tok = LlamaTokenizer.from_pretrained(sign, cache_dir=str(checkpoints_root))
    else:
        if not use_fast:
            tok = AutoTokenizer.from_pretrained(sign, padding_side=padding_side, cache_dir=str(checkpoints_root))
        else:
            tok = PreTrainedTokenizerFast.from_pretrained(sign, padding_side=padding_side, cache_dir=str(checkpoints_root))

    if model_type in ["gpt2", "e-gpt"]:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.eos_token
    if model_type in ["falcon"]:
        # tok.add_special_tokens({'pad_token': '<|pad|>'})
        tok.pad_token_id = 9#tok.eos_token
        tok.padding_side = "left"
    if 'llama' in model_type:
        tok.pad_token = "[PAD]"
        tok.padding_side = "left"
        
    return tok


def build_model(model_type, model_size, in_8bit):
    sign = build_model_signature(model_type, model_size)
    model = AutoModelForCausalLM.from_pretrained(
        sign,
        cache_dir=str(checkpoints_root),
        device_map="auto",
        load_in_8bit=in_8bit,
        token="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",
    )
    model.eval()
    return model
