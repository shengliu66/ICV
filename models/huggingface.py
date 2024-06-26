from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, LlamaTokenizer
from anchor import checkpoints_root


def build_model_signature(model_type, model_size, instruct=''):
    if model_type == 'falcon':
        return f"tiiuae/falcon-{model_size}{instruct}"
    if model_type == 'llama':
        return f"yahma/llama-{model_size}-hf"
    if model_type == 'vicuna':
        return f"lmsys/vicuna-{model_size}-v1.3"
    if model_type == 'llama-2':
        return f"meta-llama/Llama-2-{model_size}-chat-hf"
        
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
        tok.pad_token_id = 9
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
        token="",
    )
    model.eval()
    return model
