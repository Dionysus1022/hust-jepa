import argparse
import logging
import os
import random
import socket

import numpy as np
import torch
from accelerate import PartialState

from deployment.model_server.tools.batched_websocket_policy_server import BatchedWebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def main(args) -> None:
    # Flow-matching inference samples its initial action trajectory with
    # torch.randn. Seed the server before model construction so repeated
    # evaluations can use identical model initialization and sampling noise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(device)
    PartialState()
    logging.info("Inference RNG seed: %s", args.seed)

    vla = baseframework.from_pretrained(
        args.ckpt_path,
    )

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating batched server (host: %s, ip: %s)", hostname, local_ip)

    server = BatchedWebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        metadata={
            "env": "libero",
            "batching": True,
            "max_batch_size": args.max_batch_size,
            "batch_timeout_ms": args.batch_timeout_ms,
            "seed": args.seed,
        },
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
    )
    logging.info("batched server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--batch_timeout_ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    print("Waiting for VSCode attach on 0.0.0.0:10091 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    parsed_args = parser.parse_args()
    debug_enabled = os.getenv("DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if debug_enabled:
        print("DEBUGPY is enabled")
        start_debugpy_once()
    main(parsed_args)
