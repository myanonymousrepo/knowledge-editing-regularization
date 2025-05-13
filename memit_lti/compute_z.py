from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome_lti import repr_tools
from util import nethook

from .memit_hparams import MEMITLTIHyperParams
from .context import get_edit_prefix, get_edit_target


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITLTIHyperParams,
    layer: int,
    context_templates: List[str],
    dataset_name: str = None,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    target_len = len(target_ids)
    target_idxs_start = []
    target_idxs_end = []

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids): ex_len] = target_ids
        target_idxs_start.append(ex_len - len(target_ids))
        target_idxs_end.append(ex_len)

    for i in range(len(kl_prompts)):
        ex_len = input_tok["attention_mask"][i+len(rewriting_prompts)].sum()
        target_idxs_start.append(ex_len - 1)
        target_idxs_end.append(ex_len)

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context_types in context_templates
        for context in context_types
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(
                i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # target_idxs = [target_len] * len(rewriting_prompts) + loc_prompt_len
    # target_idxs_start = [target_len] * \
    #     len(rewriting_prompts) + lookup_idxs[-len(kl_prompts):]
    # target_idxs_end = [None]*len(rewriting_prompts) + \
    #     [None for idx in lookup_idxs[-len(kl_prompts):]]

    context_kl_log_probs_tar, context_kl_log_probs_mid = get_edit_target(
        model=model,
        tok=tok,
        request=request,
        hparams=hparams,
        context_prefix=get_edit_prefix(
            request=request,
            hparams=hparams,
            dataset_name=dataset_name,
        ),
        rewriting_prompts=rewriting_prompts,
        loc_prompts=kl_prompts,
        target_idxs_start=target_idxs_start,
        target_idxs_end=target_idxs_end,
        lookup_idxs=lookup_idxs,
    )

    trace_layers_mid = [
        hparams.layer_module_tmp.format(layer) for layer in hparams.midlayers
    ]
    trace_layers_final = [
        hparams.layer_module_tmp.format(layer+1) for layer in hparams.midlayers
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,),
                            requires_grad=True, device="cuda")
    else:
        delta = torch.zeros((model.config.hidden_size,),
                            requires_grad=True, device="cuda")

    target_init, kl_distr_init = None, None
    midlayer_vec, subject_vec, last_vec = None, None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, midlayer_vec, subject_vec, last_vec

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
                    tmp_repr[i, idxst:idxed, :]
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

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ] + trace_layers_mid + trace_layers_final,
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            kl_logits = torch.cat(
                [
                    logits[i, idxst:idxed, :]
                    for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                ],
                dim=0,
            )

        # print(f"MIDLAYER_VEC_SHAPE:{midlayer_vec.shape}")
        midlayer_logits = ln_f(
            midlayer_vec) @ lm_w.to(midlayer_vec.device) + lm_b.to(midlayer_vec.device)

        # kl_logits=torch.cat([midlayer_logits.squeeze(1),kl_logits],dim=0)
        kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
        mid_log_probs = torch.nn.functional.log_softmax(
            midlayer_logits.squeeze(1), dim=1)

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100,
                        rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()*hparams.nll_factor
        # nll_loss = nll_loss_each.mean()

        kl_loss = hparams.last_kl_factor * torch.nn.functional.kl_div(
            context_kl_log_probs_tar, kl_log_probs, log_target=True, reduction="batchmean"
        )

        mid_kl_loss = hparams.mid_kl_factor * torch.nn.functional.kl_div(
            context_kl_log_probs_mid, mid_log_probs, log_target=True, reduction="batchmean"
        )

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )

        device = torch.device("cuda:0")
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = kl_loss.to(device) + mid_kl_loss.to(device) + \
            weight_decay.to(device) + nll_loss.to(device)

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(mid_kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )

        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target, delta.norm().item(), target_init.norm().item()


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index(
            "subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
