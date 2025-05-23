import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome_lti.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITLTIHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
SEQ_CACHE = {} 


def apply_memit_lti_rect_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITLTIHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas, z_norms, time_compute_z, total_inserting_time= execute_memit_lti(model, tok, requests, hparams,
                           cache_template=cache_template)
    distances = {}
    distances['z_norms'] = z_norms

    with torch.no_grad():
        for w_name, (key_mat, val_mat, preservation_distance, new_edit_distance, old_edit_distance, inside_norms) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            
        
            original_weights_norm = torch.norm(w[...]).detach().cpu().item()
            k_percent = 40
            epsilon = 1e-8
            delta = torch.abs(upd_matrix / (w + epsilon))
            threshold = torch.kthvalue(delta.reshape(-1), int(delta.numel() * (100 - k_percent) / 100)).values
            with torch.no_grad():
                mask = delta >= threshold
                w[mask] += upd_matrix[mask].float()

            # w[...] += upd_matrix.float()

            if hparams.calculate_norms:
                start = time.time()
                _, svd_upd, _ = torch.linalg.svd(upd_matrix)
                svd_upd = sorted(svd_upd.detach().cpu().tolist(), reverse=True)
                
                _, svd_final, _ = torch.linalg.svd(w[...])
                svd_final = sorted(svd_final.detach().cpu().tolist(), reverse=True)
                print('svd calculation time:', time.time() - start)
            else:
                svd_final = None
                svd_upd = None

            #saving all distances
            layer = w_name.split('.')[2]
            temp_dict = {
                'preservation_distance': preservation_distance,
                'new_edit_distance': new_edit_distance,
                'old_edit_distance': old_edit_distance,
                'delta_norm': torch.norm(upd_matrix).detach().cpu().item(),
                'new_weights_norm': torch.norm(w[...]).detach().cpu().item(),
                'original_weights_norm': original_weights_norm,
                'inside_norms': inside_norms,
                #'alpha': alpha,
                'svd_final': svd_final,
                'svd_upd': svd_upd, 
            }
            distances[layer] = temp_dict

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy, distances, time_compute_z, total_inserting_time


def execute_memit_lti(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITLTIHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    z_norms = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + \
                request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    start_compute_z = time.time()
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for r_id, request in enumerate(requests):
        print(r_id)
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z, delta_norm, init_norm = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_norms[r_id] = {'delta': delta_norm, 'init_norm': init_norm, 'final_norm': cur_z.norm().item()}
            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)
    time_compute_z = time.time() - start_compute_z

    # Insert
    editing_times = []
    for i, layer in enumerate(hparams.layers):
        start_inserting_time = time.time()
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests,
                              hparams, layer, context_templates).T
        print(
            f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov, preserved_keys = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

       
        # Compute update in double precision
        layer_ks, targets, cov = (
            layer_ks.double(),
            targets.double(),
            cov.double(),
        )
        if hparams.mom2_update_weight != 1:
            cov *= hparams.mom2_update_weight

    

        #####CALCULATING UNIFIED EDITING UPDATES
        pseudo_inverse = False
        C_inv_norm = None
        D_norm = None
        D_inv_norm = None

        #calculate effective covariance matrix
        if layer in SEQ_CACHE and hparams.sequential and hparams.add_prev_edits:
            cov += SEQ_CACHE[layer].to(cov.device)

        ##Store previous sequential
        if layer not in SEQ_CACHE:
            SEQ_CACHE[layer] = (layer_ks @ layer_ks.T).cpu()
        else:
            SEQ_CACHE[layer] += (layer_ks @ layer_ks.T).cpu()


        ###NOTE - The past memory term is scaled by hparams.mom2_update_weight or sqrt of it. 
        adj_k = torch.linalg.solve(
            cov + layer_ks @ layer_ks.T,
            layer_ks,
        ).cuda()

        ###Layer distribution code
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T
        editing_times.append(time.time() - start_inserting_time)



        ##calculate_norms
        inside_norms = {
                'zs_norm' : torch.mean(torch.norm(zs, dim = 0)).detach().cpu().item(),
                'cur_zs_norm' : torch.mean(torch.norm(cur_zs, dim = 0)).detach().cpu().item(),
                'layer_ks_norm' : torch.mean(torch.norm(layer_ks, dim = 0)).detach().cpu().item(),
                'adj_norm' : torch.mean(torch.norm(adj_k , dim = 0)).detach().cpu().item(),
                'residual_norm' : torch.mean(torch.norm(resid , dim = 0)).detach().cpu().item(),
                'inside_update_norm' : torch.norm(upd_matrix).detach().cpu().item(),
                'pseudo_inverse' : pseudo_inverse,
                'C_inv_norm' : C_inv_norm,
                'D_inv_norm' : D_inv_norm,
                'D_norm' : D_norm,
                'cov' : torch.norm(cov).detach().cpu().item(),
        }
        
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()

            #calculate distances
            if hparams.calculate_objective_value:
                preservation_distance, new_edit_distance, old_edit_distance = calculate_distances(weights_copy[weight_name], weights[weight_name][...], layer_ks, zs, preserved_keys)
            else:
                preservation_distance, new_edit_distance, old_edit_distance = None, None, None
            
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
                preservation_distance, 
                new_edit_distance, 
                old_edit_distance,
                inside_norms
            )

        # Clear GPU memory
        cov = cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x = x.cpu()
            del x
        torch.cuda.empty_cache()


    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    total_editing_time = sum(editing_times)

    return deltas, z_norms,time_compute_z, total_editing_time

       
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    feature_key = (model_name, layer_name, "preserved_keys")

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat, preserved_keys = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
        COV_CACHE[feature_key] = preserved_keys

    return COV_CACHE[key].to("cuda"), COV_CACHE[feature_key] 

def calculate_distances(original_weights, new_weights, edit_keys, edit_values, preserved_keys):
    preserved_keys = preserved_keys.to("cuda")
    if original_weights.shape[0] != preserved_keys.shape[1]:
        original_weights = original_weights.T
        new_weights = new_weights.T

    W_old_k_old = preserved_keys.double() @ original_weights.double()
    W_hat_k_old = preserved_keys.double() @ new_weights.double()

    W_old_k_edits = original_weights.T.double() @ edit_keys.double()
    W_hat_k_edits = new_weights.T.double() @ edit_keys.double()
    v_edits = edit_values.double()

    preservation_distance = torch.mean(torch.norm(W_hat_k_old - W_old_k_old, dim = 1)).detach().cpu().item()
    new_edit_distance = torch.mean(torch.norm( W_hat_k_edits - v_edits, dim = 0)).detach().cpu().item()
    old_edit_distance = torch.mean(torch.norm( W_old_k_edits - v_edits, dim = 0)).detach().cpu().item()

    preserved_keys = preserved_keys.to("cpu")

    return preservation_distance, new_edit_distance, old_edit_distance

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 7)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
