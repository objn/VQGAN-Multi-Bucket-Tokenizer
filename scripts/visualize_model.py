"""Show the ViT-VQGAN layers: a per-layer summary table (type, output shape,
param count) printed straight to the console, plus the full model graph
written to TensorBoard for interactive, expandable node-by-node inspection.

Usage:
    python scripts/visualize_model.py --vqgan-checkpoint checkpoints/vqgan_last.pt
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqgan.config import VQGANTrainConfig
from vqgan.display import console
from vqgan.models import VQGAN


class _ReconOnly(torch.nn.Module):
    """VQGAN.forward() returns a VQOutput of five tensors — tracing all of them
    for add_graph routes vq_loss/token_indices/mu/log_b (training-only outputs,
    none of them part of the actual layer stack) up to a shared multi-output
    node, which is what produced the extra crossing edges in the graph view.
    Only the reconstruction matters for a layer diagram, so trace that alone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).recon


def parse_args(argv=None):
    defaults = VQGANTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vqgan-checkpoint", default=str(Path(defaults.checkpoint_dir) / "vqgan_last.pt")
    )
    parser.add_argument("--out", dest="out_dir", default="outputs/vqgan/model_graph")
    parser.add_argument("--port", type=int, default=6007)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.vqgan_checkpoint, map_location=device)
    if "model_config" not in ckpt:
        raise ValueError(
            f"{args.vqgan_checkpoint} has no model_config entry — it predates the ViT-VQGAN "
            f"rewrite and holds CNN encoder/decoder weights that cannot be loaded."
        )
    model_config = ckpt["model_config"]
    vqgan = VQGAN(**model_config).to(device)
    vqgan.load_state_dict(ckpt["vqgan"])
    vqgan.eval()  # skip the EMA codebook-update branch — only the forward pass is traced

    # The input size is not a free choice: the learned position embeddings fix
    # the token grid, so the trace has to run at the size the model was trained at.
    tile_size = model_config["image_size"]
    dummy_input = torch.zeros(1, 3, tile_size, tile_size, device=device)

    model_stats = summary(
        vqgan,
        input_data=dummy_input,
        depth=5,  # unroll encoder/quantizer/decoder down to individual layers
        col_names=("output_size", "num_params"),
        verbose=0,  # capture the table instead of letting torchinfo print it directly
    )
    console.print(str(model_stats))

    out_dir = Path(args.out_dir)
    # Start clean each run — otherwise the TensorBoard "Run" dropdown
    # accumulates one stale entry per past visualize_model.py invocation.
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    writer.add_graph(_ReconOnly(vqgan), dummy_input)
    writer.close()
    console.print(f"[green]wrote[/green] model graph to {out_dir}")

    try:
        subprocess.Popen(["tensorboard", "--logdir", str(out_dir), "--port", str(args.port)])
        console.print(f"[cyan]tensorboard:[/cyan] http://localhost:{args.port}  (open the Graphs tab)")
    except FileNotFoundError:
        console.print(
            f"[yellow]tensorboard CLI not found on PATH[/yellow] — logs are still written to {out_dir}"
        )


if __name__ == "__main__":
    main()
