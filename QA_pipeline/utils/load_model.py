from transformers import AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch
import sys


def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token

    if (
        tokenizer.sep_token_id in (None, tokenizer.vocab_size)
        and "bos_token" in special_tokens
    ):
        special_tokens["sep_token"] = special_tokens["bos_token"]

    if (
        tokenizer.pad_token_id in (None, tokenizer.vocab_size)
        and "pad_token" not in special_tokens
    ):
        if tokenizer.unk_token_id is not None:
            special_tokens["pad_token"] = tokenizer.unk_token
        else:
            special_tokens["pad_token"] = "<|pad|>"

    if (
        tokenizer.sep_token_id in (None, tokenizer.vocab_size)
        and "sep_token" not in special_tokens
    ):
        if tokenizer.bos_token_id is not None:
            special_tokens["sep_token"] = tokenizer.bos_token
        else:
            special_tokens["sep_token"] = "<|sep|>"

    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer


def fix_model(model, tokenizer, use_resize=True):
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    if use_resize:
        model.resize_token_embeddings(len(tokenizer))

    print(f"PAD ID: {model.config.pad_token_id}")
    print(f"BOS ID: {model.config.bos_token_id}")
    print(f"EOS ID: {model.config.eos_token_id}")

    return model


def load_model(model_config):
    tokenizer = LlamaTokenizer.from_pretrained(model_config["name"], eos_token="</s>")
    tokenizer.padding_side = model_config["padding_side"]

    tokenizer = fix_tokenizer(tokenizer)

    if model_config["torch_dtype"] == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        load_in_8bit=model_config["load_in_8bit"],
        torch_dtype=torch_dtype,
        device_map=model_config["device_map"],
    )
    model = fix_model(model, tokenizer, not model_config["load_in_8bit"]) ####!!!
    if model_config.get("peft_model_id"):
        model = PeftModel.from_pretrained(
            model, model_config["peft_model_id"], torch_dtype=torch_dtype, device_map=model_config["device_map"] ##
        )

    return model, tokenizer
