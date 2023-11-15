import json
import logging
import random
import re
from collections import defaultdict

import torch
import numpy as np
import datasets

from anchor import hf_datasets_root
from tasks.loader import TokenizedForStyleRightPad
from utils.rng_ctx import RandomContext, EmptyContext
from utils.pca import PCA
from utils.context_manager import modified_forward_context_manager, traced_forward_context_manager
from utils.latent_shifter import LatentShifter
logger = logging.getLogger("task")


class BaseProbInference:
    def __init__(self, prompt_version):
        if prompt_version == "default":
            self.prompt_version = self.default_prompt_version()
        else:
            self.prompt_version = prompt_version

        self.raw_data_sample = None
        self.raw_data_dev = None

        self.can_be_stratified = False
        self.num_base_shot = 1

        self._rng_context = EmptyContext()

        self._cached_prefix = None
        self._cached_ex_list = None
        self._cahced_selected_exemplar = None
        self.shuffled_mapping = None

    def default_prompt_version(self):
        raise NotImplementedError

    def set_seed(self, seed):
        self._rng_context = RandomContext(seed=seed)

    def dataset_signature(self):
        # {
        #      "result":  (dataset_name, subset, split),  # which produce the final result
        #      "sample": (dataset_name, subset, split),  # which we sample ICL few-shot examples
        # }
        raise NotImplementedError

    def dataset_part(self, part):
        return self.dataset_signature()[part]

    def dataset_preprocess(self, raw_data):
        raise NotImplementedError

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        raise NotImplementedError

    def paralell_style_promptify(self, query):
        raise NotImplementedError

    def shuffle_exemplars(self):
        prefix = self._cached_prefix
        ex_list = self._cached_ex_list

        ex_list_with_idx = list(enumerate(ex_list))
        with self._rng_context:
            random.shuffle(ex_list_with_idx)

        indices, ex_list = zip(*ex_list_with_idx)
        self.shuffled_mapping = indices

        return self.build_exemplar_from_examples(prefix, ex_list)

    def random_selected_exemplars(self, num_shots, prefix = ""):

        with self._rng_context:
            num_shots = min(len(self.raw_data_sample), num_shots)
            sampled = random.sample(self.raw_data_sample, num_shots)

        self._cahced_selected_exemplar = sampled

        ex_list = [e["query"] for e in sampled]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list
        return self.build_exemplar_from_examples(prefix, ex_list)

    def stratified_sampling(self, num_k_shots):
        num_shots = self.num_base_shot * num_k_shots

        if not self.can_be_stratified:
            logger.info("Cannot be stratified, fallback to random selection.")
            return self.random_selected_exemplars(num_shots)

        prefix = ""

        ans_set = set(e["answer_idx"] for e in self.raw_data_sample)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sample):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = num_shots // len(ans_set)
        residual = num_shots - per_label * len(ans_set)

        selected_ids = []
        with self._rng_context:
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sample))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        selected_exemplar = [self.raw_data_sample[i] for i in selected_ids]
        self._cahced_selected_exemplar = selected_exemplar
        ex_list = [e["query"] for e in selected_exemplar]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list
        return self.build_exemplar_from_examples(prefix, ex_list)

    def build_exemplar_from_examples(self, prefix, ex_list):
        s = prefix
        if len(s):
            s += self.exemplar_seperator()

        for query in ex_list:
            _, line = self.paralell_style_promptify(query)  # query, <query_with_answer>
            s += line + self.exemplar_seperator()
        return s

    def dataset_file_path(self, part):
        dataset_name, subset, split = self.dataset_part(part)
        dumped_folder = hf_datasets_root.joinpath("dumped")
        if not dumped_folder.exists():
            dumped_folder.mkdir(parents=True)

        if part == "sample":
            split = 'train' 
        if part == "result":
            split = 'test' 

        file_name = f"{dataset_name}-{subset}-{split}.jsonl"
        file_name = re.sub(r"[^\w_. -]", "_", file_name)
        return dumped_folder.joinpath(file_name)

    def do_load_part(self, part):
        f_path = self.dataset_file_path(part)
        print(f_path)
        if not f_path.exists():
            self.not_exist_download(part)
            return self.do_load_part(part)  # call once more
        else:
            with f_path.open("r") as f:
                raw_data = [json.loads(line) for line in f]
            data = self.dataset_preprocess(raw_data)
            logger.info(f"Data loaded: {part}.")
            return data

    def do_load(self):
        self.raw_data_sample = self.do_load_part("sample")
        self.raw_data_result = self.do_load_part("result")

    def not_exist_download(self, part):
        f_path = self.dataset_file_path(part)
        logger.info(f"{f_path} not exist, download from huggingface datasets hub...")

        dataset_name, subset, split = self.dataset_part(part)
        data = self.do_download(dataset_name, subset, split=split, cache_dir=str(hf_datasets_root))

        if part == "sample":
            data = data.train_test_split(test_size=0.4)['train']
        if part == "result":
            data = data.train_test_split(test_size=0.4)['test']

        data.to_json(f_path)
        logger.info(f"... success, saved at: {f_path}")

    @staticmethod
    def do_download(dataset_name, subset, split, cache_dir):
        raw_data = datasets.load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
        logger.info("Download success.")
        return raw_data

    def mk_result_dataset(self, tokenizer, no_padding=False, prefix=''):
        return TokenizedForStyleRightPad(self.raw_data_result, tokenizer, self.paralell_style_promptify, no_padding=no_padding, prefix=prefix)

    def mk_test_dataset(self, tokenzier):
        return self.mk_result_dataset(tokenzier)


    def mk_dev_dataset(self, tokenizer):
        sample_size = len(self.raw_data_result)

        ans_set = set(e["answer_idx"] for e in self.raw_data_sample)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sample):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = sample_size // len(ans_set)
        residual = sample_size - per_label * len(ans_set)

        selected_ids = []
        with self._rng_context:
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sample))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        self.raw_data_dev = [self.raw_data_sample[i] for i in selected_ids]
        return TokenizedForStyleRightPad(self.raw_data_dev, tokenizer, self.paralell_style_promptify)

    def mk_finetune_dataset(self, tokenizer, mode = 'ft'):
        selected_exemplar = self._cahced_selected_exemplar
        assert (selected_exemplar != None), "No demonstration is selected yet, run stratified_sampling first! \n"
        return TokenizedForStyleRightPad(selected_exemplar, tokenizer, self.paralell_style_promptify, mode=mode)

    def mk_result_dataset_with_demostration(self, tokenizer, exemplar_str, no_padding=False):
        def add_demostration(query, return_reference = False, Instruction = ''):
            if return_reference:
                with_query, with_query_and_paraphrase, references = self.paralell_style_promptify(query, return_reference=return_reference, Instruction=Instruction)
                with_query = with_query.replace(Instruction,"")
                with_query_and_paraphrase = with_query_and_paraphrase.replace(Instruction,"")
                return f"{exemplar_str}{with_query}", f"{exemplar_str}{with_query_and_paraphrase}", references
            else:
                with_query, with_query_and_paraphrase = self.paralell_style_promptify(query, return_reference=return_reference, Instruction=Instruction)
                with_query = with_query.replace(Instruction,"")
                with_query_and_paraphrase = with_query_and_paraphrase.replace(Instruction,"")
                return f"{exemplar_str}{with_query}", f"{exemplar_str}{with_query_and_paraphrase}"

        return TokenizedForStyleRightPad(self.raw_data_result, tokenizer, add_demostration, no_padding=no_padding)

    @staticmethod
    def modified_forward(model, inputs, forward_modifiers = ()):
        context_manager = modified_forward_context_manager(model, forward_modifiers=forward_modifiers)
        input_ids = torch.tensor(inputs['input_ids'])
        attention_mask = torch.tensor(inputs['attention_mask'])
        with context_manager:
            outputs = model(
                input_ids=input_ids.unsqueeze(0).cuda(),
                attention_mask=attention_mask.unsqueeze(0).cuda(),
            )

        return outputs

    @staticmethod
    def traced_forward(model, inputs, forward_modifiers = (), with_submodules=False):
        context_manager, forward_trace = traced_forward_context_manager(model, with_submodules=with_submodules)
        with context_manager:
            outputs = BaseProbInference.modified_forward(
                model,
                inputs=inputs,
                forward_modifiers=forward_modifiers,
            )

        return outputs, forward_trace

    @staticmethod
    def get_latentstates(model, inputs):
        h_all = []
        for example_id in range(len(inputs)):
            latents_for_all_styles= []
            for style_id in range(len(inputs[example_id])):
                _, forward_trace = BaseProbInference.traced_forward(model, inputs[example_id][style_id], with_submodules=False)
                task_latents = forward_trace.residual_stream.hidden[:, :, -1, :] # get last token
                task_latents = task_latents[:, 1:]  # the first one is the embedding layer (num_data, num_layers, hidden_size)
                latents_for_all_styles.append(task_latents)
            h_all.append(tuple(latents_for_all_styles))
        return h_all

    @staticmethod
    def get_icv(model, inputs, rank=1):
        hidden_states = BaseProbInference.get_latentstates(model, inputs)
        _, num_layers, hidden_dim = hidden_states[0][0].size()

        hidden_states_all = []
        num_demonstration = len(hidden_states)
        for demonstration_id in range(num_demonstration):
            h = hidden_states[demonstration_id][0].flatten() - hidden_states[demonstration_id][1].flatten()
            hidden_states_all.append(h)
        
        fit_data = torch.stack(hidden_states_all)        
        pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())

        direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0)

        return direction.view(num_layers, hidden_dim)   