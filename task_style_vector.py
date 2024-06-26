import gc
import json
import logging
import os
import textwrap


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anchor import logger_root
from common import setup_env, mk_parser, AdvantageLogger
from models import build_model_signature, build_tokenizer, build_model
from tasks import load_task
from utils.logger import setup_logger, tabular_pretty_print
from utils.tools import ensure_folder
from utils.pca import PCA
from utils.llm_layers import add_icv_layers, remove_icv_layers
import numpy as np
import pdb
        
logger = logging.getLogger("task")

if __name__ == "__main__":
    parser = mk_parser()
    args = parser.parse_args()

    logger_root = logger_root.joinpath("main")
    dataset_name = args.dataset
    logger_folder = logger_root.joinpath(dataset_name)

    task_name = f"seed{args.seed}"
    task_name += f"_{args.prompt_version}"
    task_name += f"_{args.model_type}_{args.model_size}"
    task_name += f"_{args.exemplar_method}{'' if args.exemplar_method == 'written' else args.num_k_shots}"
    task_name += f"_icvstrength{args.lam}"
    
    setup_env(gpu_s=args.gpus, seed=args.seed)
    ensure_folder(logger_folder, parents=True)
    setup_logger(
        logger_folder,
        log_file_name=f"{task_name}.log",
        console_output=not args.no_console,
    )

    logger.info(f"Task Prepared: {task_name}")
    logger.info(f"\tDataset: {dataset_name}")
    logger.info(f"\tLogger save at {logger_folder}")

    # 1. load model, tokenizer
    model_signature = build_model_signature(args.model_type, args.model_size)

    padding_side = 'right'

    tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side=padding_side)

    model = build_model(args.model_type, args.model_size, args.in_8bit)
    torch.autograd.set_grad_enabled(False)
    logger.info(f"Model loaded: {model_signature}")

    # 2. load dataset (with demonstrations)
    TaskHandler = load_task(dataset_name)
    task_agent = TaskHandler(args.prompt_version)
    task_agent.set_seed(args.seed)
    task_agent.do_load()

    dataset = task_agent.mk_result_dataset(tokenizer, no_padding=True, prefix='Please paraphrase the following sentence.\n ')

    if args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    elif args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_k_shots, prefix='Please paraphrase the following sentence.\n\n')
    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_k_shots)
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    text_width = 168
    exemplar_showcase = [["Line", "Text"]]
    for line_idx, line in enumerate(exemplar_str.split("\n")):
        if len(line) > text_width:
            splitted_lines = textwrap.wrap(line, text_width)
            exemplar_showcase.append([str(line_idx + 1), splitted_lines[0]])
            for remained in splitted_lines[1:]:
                exemplar_showcase.append(["", remained])
        else:
            exemplar_showcase.append([str(line_idx + 1), line])

    exemplar_showcase[-1][-1] += "<query starts from here>"
    for line in tabular_pretty_print(exemplar_showcase):
        logger.info(line)


    icv, _ = task_agent.obtain_icv(
        model, dataset.tokenize_each_demonstration(
            task_agent._cached_ex_list.copy(), prefix=("", "")
            ), rank=1
        )

    icv = icv[1:]

    logger.info(f"Add in-context vectors: {args.batch_size}")

    logger.info(f"Selected batch_size: {args.batch_size}")

    loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=2)

    logger.info("Running ...")

    add_icv_layers(model, torch.stack([icv],dim=1).cuda(), [args.lam])



    if 'llama' in args.model_type:
        gen_args = {
            'temperature': 0.45,
            'do_sample': True,
            'top_k': 0,
            'top_p': 1.0,
            'eos_token_id': [1642, 13492, 26036, 29908,tokenizer.encode('.10')[-1]]
        }
    elif 'falcon' in args.model_type:
        gen_args = {
            'do_sample': False,
            'num_beams': 10,
            'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166]
        }
    else:
        gen_args = {}

    with torch.no_grad():
        ans_file = open(logger_folder.joinpath(task_name + '.json') , 'w')
        for batch_input in tqdm(loader, desc=f"Evaluation"):
            batch_input_ids = batch_input[0]
            print(tokenizer.batch_decode(batch_input_ids))
            batch_masks = batch_input[1]
            batch_reference = batch_input[2]
            # try:

            generation_output = model.generate(
                input_ids=batch_input_ids.cuda(),
                attention_mask=batch_masks.cuda(),
                max_new_tokens=32,
                **gen_args,
            )

            generation_output = tokenizer.decode(generation_output[0][len(batch_input_ids[0]):]).replace("\n","").replace("{","").replace("}","").replace('"','').strip('".').replace(',,','').replace('original','').replace('Original','').split('rewritten')[0].split('revised')[0].replace('10','').split('.')[0]

            logger.info(f'generation: {generation_output}, gold: {batch_reference[0]} \n')
            ans_file.write(json.dumps({"generation": generation_output,
                                        "gold": batch_reference[0],
                                       }) + "\n")
            ans_file.flush()
        ans_file.close()

    remove_icv_layers(model)

    
