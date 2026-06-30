# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Precompute UMT5 text embeddings for the Wan2.2 S2V inferencer.

Run this ONCE per fixed prompt (e.g. per anchor/variant). The resulting
embedding file can then be fed to `generate.py --no_text_encoder
--prompt_embeds <file>`, so the ~6 GB UMT5 text encoder never has to be
downloaded or loaded on the GPU render pod.

The saved tensor is the raw text-encoder output of shape (L, 4096); the DiT
applies its own 4096->dim projection internally, so no further processing is
needed at inference time.

Example:
    python precompute_text_embeds.py \
        --task s2v-14B \
        --ckpt_dir ./Wan2.2-S2V-14B \
        --prompt "A news anchor reads the headlines." \
        --output anchor_erica.pt
"""
import argparse
import logging
import os

import torch

from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel


def main():
    parser = argparse.ArgumentParser(
        description="Precompute UMT5 text embeddings for Wan2.2 S2V.")
    parser.add_argument(
        "--task",
        type=str,
        default="s2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="Task whose text-encoder config to use.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory containing the UMT5 encoder + tokenizer.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Positive prompt to encode.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Optional negative prompt. Defaults to the task's sample "
        "negative prompt. Use --no_negative to skip it.")
    parser.add_argument(
        "--no_negative",
        action="store_true",
        default=False,
        help="Do not produce a negative-prompt embedding.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the positive embedding (.pt or .safetensors). "
        "The negative embedding is written alongside with a '.neg' suffix.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the encoder on.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = WAN_CONFIGS[args.task]
    device = torch.device(args.device)

    encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )

    def encode(text):
        # T5EncoderModel returns a list of one (L, 4096) tensor.
        return encoder([text], device)[0].cpu().contiguous()

    def save(tensor, path):
        if path.endswith(".safetensors"):
            from safetensors.torch import save_file
            save_file({"prompt_embeds": tensor}, path)
        else:
            torch.save(tensor, path)
        logging.info(f"Saved {tuple(tensor.shape)} -> {path}")

    pos = encode(args.prompt)
    save(pos, args.output)

    if not args.no_negative:
        neg_prompt = args.negative_prompt
        if neg_prompt is None:
            neg_prompt = cfg.sample_neg_prompt
        neg = encode(neg_prompt)
        root, ext = os.path.splitext(args.output)
        neg_path = f"{root}.neg{ext}"
        save(neg, neg_path)


if __name__ == "__main__":
    main()
