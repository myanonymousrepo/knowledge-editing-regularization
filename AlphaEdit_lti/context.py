#from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import json
from torch.utils.data import Dataset
from .AlphaEdit_hparams_lti import AlphaEditLTIHyperParams
import os
from random import shuffle
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from util.globals import *
from util import nethook
import torch


# def get_edit_prefix(
#     request: Dict,
#     hparams: MEMITHyperParams,
#     dataset_name: str,
# ) -> list:

#     assert dataset_name is not None
#     sentence_model = SentenceTransformer(
#         hparams.sentence_model_name).to("cuda")

#     safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
#     with open(f'{EMBEDDING_DIR}/{safe_model_name}_{dataset_name}.pkl', "rb") as fIn:
#         stored_data = pickle.load(fIn)

#     new_fact = request['prompt'].format(
#         request['subject']) + request['target_new']['str']
#     query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
#     query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
#         query_sentence, show_progress_bar=False)).unsqueeze(0).to("cuda"))

#     combined_results = []

#     # 循环处理每个案例
#     for case in stored_data:
#         stored_sentences = stored_data[case]['sentences']
#         stored_embeddings = torch.tensor(
#             stored_data[case]['embeddings']).to("cuda")
#         stored_embeddings = util.normalize_embeddings(stored_embeddings)

#         # 每个案例中的嵌入检索并获取top-k结果
#         hits = util.semantic_search(
#             query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.top_k)
#         assert len(hits) == 1
#         hit = hits[0]
#         icl_examples = [stored_sentences[hit[k]["corpus_id"]]
#                         for k in range(len(hit))]

#         # 将结果添加至总列表
#         combined_results.extend(icl_examples)

#     shuffle(combined_results)
#     # combined_results.append(f'New Fact: {new_fact}\nPrompt: {new_fact}\n\n')
#     combined_results.append(f'New Fact: {new_fact}\nPrompt: ')

#     return ''.join(combined_results)


def get_edit_prefix(
    request: Dict,
    hparams: AlphaEditLTIHyperParams,
    dataset_name: str,
) -> str:
    new_fact = request['prompt'].format(
        request['subject']) + request['target_new']['str']

    context_demo = f"Imagine that {new_fact}\n"

    return context_demo


# def get_edit_prefix(
#     request: Dict,
#     hparams: MEMITHyperParams,
#     dataset_name: str,
# ) -> str:

#     assert dataset_name is not None
#     sentence_model = SentenceTransformer(
#         hparams.sentence_model_name).to("cuda")

#     safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
#     with open(f'{EMBEDDING_DIR}/'
#               f'{safe_model_name}_{dataset_name}.pkl', "rb") as fIn:
#         stored_data = pickle.load(fIn)
#         stored_sentences = stored_data['sentences']
#         stored_embeddings = stored_data['embeddings']
#     stored_embeddings = torch.tensor(stored_embeddings).to("cuda")
#     stored_embeddings = util.normalize_embeddings(stored_embeddings)

#     new_fact = request['prompt'].format(request['subject']) + request['target_new']['str']
#     query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
#     query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
#         query_sentence, show_progress_bar=False)).unsqueeze(0).to("cuda"))

#     hits = util.semantic_search(
#         query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.top_k)
#     assert len(hits) == 1
#     hit = hits[0]
#     icl_examples = [stored_sentences[hit[k]["corpus_id"]]
#                     for k in range(len(hit))]
#     icl_examples.append(f'New Fact: {new_fact}\nPrompt: ')

#     combined_string = ''.join(icl_examples)

#     return combined_string


def get_edit_target(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditLTIHyperParams,
    context_prefix: str,
    rewriting_prompts: list,
    loc_prompts: list,
    target_idxs_start: list,
    target_idxs_end: list,
    lookup_idxs: list,
) -> (torch.Tensor, torch.Tensor):

    print(context_prefix)

    rewriting_prompts = [context_prefix+prompt for prompt in rewriting_prompts]
    all_prompts = rewriting_prompts + loc_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    trace_layers_mid = [
        hparams.layer_module_tmp.format(layer) for layer in hparams.midlayers
    ]
    trace_layers_final = [
        hparams.layer_module_tmp.format(layer+1) for layer in hparams.midlayers
    ]

    midlayer_vec, subject_vec, last_vec = None, None, None

    def get_output_fn(cur_out, cur_layer):
        nonlocal midlayer_vec, subject_vec, last_vec

        if cur_layer in trace_layers_mid:
            tmp_repr = cur_out[0]
            subject_vec = torch.stack(
                [
                    tmp_repr[i, idx, :]
                    for i, idx in enumerate(lookup_idxs)
                ],
                dim=0
            ).unsqueeze(1)

        if cur_layer in trace_layers_final:
            tmp_repr = cur_out[0]
            last_vec = torch.cat(
                [
                    tmp_repr[i, -idxst:idxed, :]
                    for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                ],
                dim=0
            ).unsqueeze(1)

        # 根据 hparams.constr_pos 决定返回哪个向量或组合
        if hparams.constr_pos == "subject":
            midlayer_vec = subject_vec
        elif hparams.constr_pos == "last":
            midlayer_vec = last_vec
        elif hparams.constr_pos == "all":
            if last_vec is not None:
                midlayer_vec = torch.cat([subject_vec, last_vec], dim=0)
        else:
            raise ValueError(
                f"Unsupported constr_pos: {hparams.constr_pos}")

        return cur_out

    with torch.no_grad():
        with nethook.TraceDict(
            module=model,
            layers=trace_layers_mid + trace_layers_final,
            retain_input=False,
            retain_output=True,
            edit_output=get_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            kl_logits = torch.cat(
                [
                    logits[i, idxst:idxed, :]
                    for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                ],
                dim=0,
            )

            print(f"MIDLAYER_VEC_SHAPE:{midlayer_vec.shape}")
            midlayer_logits = ln_f(
                midlayer_vec) @ lm_w.to(midlayer_vec.device) + lm_b.to(midlayer_vec.device)

            print(target_idxs_start)
            print(target_idxs_end)
            print(f"TARGETSHAPE:{kl_logits.shape}")

            mid_kl_log_probs = torch.nn.functional.log_softmax(
                midlayer_logits.squeeze(1), dim=1).detach().clone()
            target_kl_log_probs = torch.nn.functional.log_softmax(
                kl_logits, dim=1).detach().clone()

    return target_kl_log_probs, mid_kl_log_probs
