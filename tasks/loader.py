import logging

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger("task")


class TokenizedForStyleRightPad(Dataset):
    def __init__(self, data, tok: PreTrainedTokenizer, prompt_fn, mode = 'eval', no_padding=False, prefix=''):
        # data: [query: str, choices: list(str)]
        self.tok = tok
        self.prompt_fn = prompt_fn
        self.references = None
        self.max_length = self._find_max_length(data, mode=mode)
        if mode == 'ft':
            self.data = self._build_ft_data(data)
        elif mode == 'eval':
            self.data, self.references = self._build_eval_data(data, no_padding=no_padding, prefix=prefix)
        else:
            raise NotImplementedError
        logger.info(f"Tokenization finished: {len(self.data)}, max_length={self.max_length}")

    def _find_max_length(self, data, mode=eval):
        max_len = 0

        def tok_len(t):
            return len(self.tok.encode(t))

        for ex in tqdm(data, desc="Data preprocessing(1/2)"):
            query = ex["query"]
            if mode == 'eval':
                len_query = len(self.prompt_fn(query)[0])
            elif mode == 'ft':
                len_query = len(self.prompt_fn(query)[1])
            else:
                raise NotImplementedError
            max_len = max(max_len, len_query)
        return max_len

    def _build_eval_data(self, data, no_padding=False, prefix=''):
        processed = []
        references = []
        for ex in tqdm(data, desc="Data preprocessing(2/2)"):
            query = ex["query"]
            processed_input = self.prompt_fn(query, return_reference = True, Instruction = prefix)
            t_query, t_full, t_reference = processed_input
            processed_input = self.tokenize(t_full, t_query, no_padding=no_padding)
            processed.append(processed_input)
            references.append(t_reference)

        logger.info("Style dataset: finish!")
        return processed, references

    def _build_ft_data(self, data):
        processed = []
        for ex in tqdm(data, desc="Data preprocessing(2/2)"):
            query = ex["query"]
            processed_input = self.prompt_fn(query)
            t_query, t_full = processed_input
            processed_input = self.tokenize(t_query, t_full)
            processed.append(processed_input)

        logger.info("Finetuning dataset: finish!")
        return processed

    def tokenize_demonstration(self, demonstration):
        e = self.tok(demonstration)
        return torch.LongTensor(e["input_ids"]), torch.LongTensor(e["attention_mask"])  # no padding



    def tokenize_each_demonstration(self, demonstration_list, dataset_name=None, prefix = None):
        special_characters = [
            "~", " ~", "~ ", "!", " !", "! ", "@", " @", "@ ", "#", " #", "# ", 
            "$", " $", "$ ", "%", " %", "% ", "^", " ^", "^ ", "&", " &", "& ", 
            "*", " *", "* ", "(", " (", "( ", ")", " )", ") ", "_", " _", "_ ", 
            "+", " +", "+ ", "`", " `", "` ", "-", " -", "- ", "=", " =", "= ", 
            "{", " {", "{ ", "}", " }", "} ", "[", " [", "[ ", "]", " ]", "] ", 
            "|", " |", "| ", "\\", " \\", "\\ ", ":", " :", ": ", ";", " ;", "; ", 
            "\"", " \"", "\" ", "'", " '", "' ", "<", " <", "< ", ">", " >", "> ", 
            ",", " ,", ", ", ".", " .", ". ", "?", " ?", "? ", "/", " /", "/ "
        ]

        def strip_special_characters(input_string):
            for char in special_characters:
                input_string = input_string.replace(char.strip(), '')
            return input_string.strip()

        tokenized_demonstration_list = []
        for exp_id in range(len(demonstration_list)):
            if prefix is not None:
                demonstration_list[exp_id] = (prefix[0] + strip_special_characters(demonstration_list[exp_id][0]), prefix[1] + strip_special_characters(demonstration_list[exp_id][1]))
            else:
                demonstration_list[exp_id] = (strip_special_characters(demonstration_list[exp_id][0]), strip_special_characters(demonstration_list[exp_id][1]))
            e_original = self.tok(demonstration_list[exp_id][0]) 
            e_rewrite = self.tok(demonstration_list[exp_id][1])
            tokenized_demonstration_list.append((e_original, e_rewrite)) 
        return tokenized_demonstration_list

    def tokenize(self, only_query, full_text, no_padding = False):
        tok_only_query = self.tok(only_query, add_special_tokens=False)
        tok_full_no_padding = self.tok(full_text, add_special_tokens=False)
        tok_full = self.tok(
            full_text,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=False,
        )  # <pad> is not a special token

        if no_padding: 
            e = {
            "input_ids": tok_full_no_padding.input_ids,
            "attention_mask": tok_full_no_padding.attention_mask,
            }
        else:
            e = {
                "input_ids": tok_full.input_ids,
                "attention_mask": tok_full.attention_mask,
            }

        return e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        es = self.data[idx]

        if self.references:
            return torch.LongTensor(es["input_ids"]), torch.LongTensor(es["attention_mask"]), self.references[idx]
        else:
            return es


if __name__ == "__main__":
    from anchor import hf_datasets_root

    import datasets

    csqa1 = datasets.load_dataset("commonsense_qa", cache_dir=str(hf_datasets_root), split="validation")
