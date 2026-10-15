# SPDX-License-Identifier: Apache-2.0
"""Generate the GenEval prompt set through a running HunyuanImage-3.0 server.

GenEval (https://github.com/djghosh13/geneval) scores object-focused compositional
prompts -- single and two objects, counting, colors, position, attribute binding -- with
an object detector over the generated images. This script writes its expected layout
from ``prompts/evaluation_metadata.jsonl``::

    <outdir>/<prompt index, 5 digits>/metadata.jsonl
    <outdir>/<prompt index, 5 digits>/samples/<sample, 4 digits>.png

one request per image to ``serve.sh``'s ``/v1/chat/completions`` (1024 x 1024, the
model's default 50 steps and CFG 2.5, seed = ``--seed`` + sample index). Each prompt
folder also records what its images were generated with (``generation.json``: the
prompt metadata, seed, size); an interrupted run resumes by skipping images already on
disk, and refuses a folder whose record differs -- a changed prompt set, seed or size
needs a new ``--outdir``.

    bash examples/hunyuan_image3/serve.sh /workspace/models/HunyuanImage-3.0-Instruct 8091 &
    python examples/hunyuan_image3/geneval_generate.py \\
        --metadata geneval/prompts/evaluation_metadata.jsonl --outdir out/geneval --samples 1
"""

import argparse
import base64
import json
import os
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--metadata", required=True, help="GenEval prompts/evaluation_metadata.jsonl")
parser.add_argument("--outdir", required=True)
parser.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
parser.add_argument("--samples", type=int, default=4, help="images per prompt (GenEval uses 4)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--size", type=int, default=1024)
args = parser.parse_args()


def generate(prompt: str, seed: int) -> bytes:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": {"height": args.size, "width": args.size, "seed": seed},
    }
    request = urllib.request.Request(
        args.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.load(response)
    url = reply["choices"][0]["message"]["content"][0]["image_url"]["url"]
    return base64.b64decode(url.split(",", 1)[1])


def main() -> None:
    with open(args.metadata) as handle:
        prompts = [json.loads(line) for line in handle if line.strip()]
    start, done = time.perf_counter(), 0
    for index, metadata in enumerate(prompts):
        folder = os.path.join(args.outdir, f"{index:05d}")
        record = {"metadata": metadata, "seed": args.seed, "size": args.size}
        record_path = os.path.join(folder, "generation.json")
        if os.path.isdir(folder):
            previous = json.load(open(record_path)) if os.path.isfile(record_path) else None
            if previous != record:
                raise SystemExit(
                    f"{folder} was generated with {previous}, not {record}: use a new --outdir"
                )
        os.makedirs(os.path.join(folder, "samples"), exist_ok=True)
        with open(record_path, "w") as handle:
            json.dump(record, handle)
        with open(os.path.join(folder, "metadata.jsonl"), "w") as handle:
            handle.write(json.dumps(metadata) + "\n")
        for sample in range(args.samples):
            path = os.path.join(folder, "samples", f"{sample:04d}.png")
            if os.path.isfile(path):
                continue
            image = generate(metadata["prompt"], args.seed + sample)
            with open(path + ".tmp", "wb") as handle:
                handle.write(image)
            os.replace(path + ".tmp", path)
            done += 1
        elapsed = time.perf_counter() - start
        print(f"[{index + 1}/{len(prompts)}] {metadata['tag']}: {metadata['prompt']!r} "
              f"({done} new images, {elapsed / max(done, 1):.1f} s each)", flush=True)


if __name__ == "__main__":
    main()
