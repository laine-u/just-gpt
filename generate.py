import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from core.inference.loader import load_model
from core.inference.generation import generate_tokens
from core.inference.sampling import validate_sampling
from core.inference.records import load_prompts, save_output
from core.misc import non_neg_int


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate text from a pretrained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Checkpoint file, eg. checkpoints/best.pt"
    )
    parser.add_argument("--max-new-tokens", type=non_neg_int, default=100)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", type=Path)

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--prompts-file", type=Path)

    sampling = parser.add_argument_group("Sampling")
    sampling.add_argument("--do-sample", action="store_true")
    sampling.add_argument("--temperature", type=float, default=1.0)
    sampling.add_argument("--top-k", type=non_neg_int, default=0)
    sampling.add_argument("--top-p", type=float, default=1.0)
    sampling.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    try:
        validate_sampling(
            args.temperature,
            args.top_k,
            args.top_p,
        )

        if args.prompts_file is not None:
            prompts = load_prompts(args.prompts_file)
        else:
            prompts = [{
                "id": "single",
                "category": "single",
                "prompt": args.prompt,
            }]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    model, tokenizer, checkpoint_info = load_model(args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "seed_policy": "reset_same_seed_for_each_prompt",
        "context_policy": "recompute_last_context_window",
    }

    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checkpoint": checkpoint_info,
        "tokenizer": tokenizer.name,
        "environment": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "dtype": str(dtype),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
        },
        "prompts_file": (
            str(args.prompts_file.resolve())
            if args.prompts_file is not None
            else None
        ),
        "num_prompts": len(prompts),
        "num_completed": 0,
        "generation_config": generation_config,
        "results": [],
    }

    print(
        f"[model] checkpoint_step={checkpoint_info['step']}, "
        f"device={device}, prompts={len(prompts)}"
    )

    if args.output is not None:
        save_output(report, args.output)

    for index, item in enumerate(prompts, start=1):
        prompt_ids = tokenizer.encode_ordinary(item["prompt"])
        generated_ids, finish_reason = generate_tokens(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=(None if args.ignore_eos else tokenizer.eot_token),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )

        # no need to display EOS
        visible_ids = generated_ids
        if finish_reason == "eos":
            visible_ids = generated_ids[:-1]

        result = {
            "id": item["id"],
            "category": item.get("category", "uncategorized"),
            "prompt": item["prompt"],
            "prompt_ids": prompt_ids,
            "generated_ids": generated_ids,
            "generated_text": tokenizer.decode(visible_ids),
            "full_text": tokenizer.decode(prompt_ids + visible_ids),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids),
            "finish_reason": finish_reason,
            "sampling_seed": args.seed if args.do_sample else None,
        }

        report["results"].append(result)
        report["num_completed"] = index

        if index == len(prompts):
            report["status"] = "completed"
            report["completed_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

        if args.output is not None:
            save_output(report, args.output)

        print(f"\n[{index}/{len(prompts)}] {item['id']}")
        print(result["full_text"])
        print(
            f"[finish] reason={finish_reason}, "
            f"prompt_tokens={len(prompt_ids)}, "
            f"generated_tokens={len(generated_ids)}"
        )

    if args.output is not None:
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()