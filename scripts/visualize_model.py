"""Show the VQGAN's layers: a per-layer summary table (type, output shape,
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

from vqgan.config import PreprocessConfig, VQGANTrainConfig
from vqgan.display import console
from vqgan.models import VQGAN


class _ReconOnly(torch.nn.Module):
    """VQGAN.forward() returns (x_recon, vq_loss, token_indices) — tracing
    all three for add_graph routes vq_loss/token_indices (a training-only
    scalar and a discrete index tensor, neither part of the actual layer
    stack) all the way up to a shared multi-output node, which is what was
    producing the extra crossing edges in the graph view. Only x_recon
    matters for a layer diagram, so trace that alone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x_recon, _, _ = self.model(x)
        return x_recon


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqgan-checkpoint", default=str(Path(VQGANTrainConfig().checkpoint_dir) / "vqgan_last.pt"))
    parser.add_argument("--canvas-size", type=int, default=PreprocessConfig().canvas_size)
    parser.add_argument("--out", dest="out_dir", default="outputs/vqgan/model_graph")
    parser.add_argument("--port", type=int, default=6007)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.vqgan_checkpoint, map_location=device)
    vqgan = VQGAN(latent_dim=ckpt["latent_dim"], num_embeddings=ckpt["num_embeddings"]).to(device)
    vqgan.load_state_dict(ckpt["vqgan"])
    vqgan.eval()  # skip the EMA codebook-update branch — only the forward pass is traced

    dummy_input = torch.zeros(1, 3, args.canvas_size, args.canvas_size, device=device)

    model_stats = summary(
        vqgan,
        input_data=dummy_input,
        depth=5,  # unroll encoder/quantizer/decoder down to individual conv/norm layers
        col_names=("output_size", "num_params"),
        verbose=0,  # capture the table instead of letting torchinfo print it directly
    )
    console.print(str(model_stats))

    out_dir = Path(args.out_dir)
    # Start clean each run — otherwise TensorBoard's "Run" dropdown
    # accumulates one stale entry per past visualize_model.py invocation.
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    writer.add_graph(_ReconOnly(vqgan), dummy_input)
    writer.close()
    console.print(f"[green]wrote[/green] model graph to {out_dir}")

    try:
        subprocess.Popen(["tensorboard", "--logdir", str(out_dir), "--port", str(args.port)])
        console.print(f"[cyan]tensorboard:[/cyan] http://localhost:{args.port}  (open the 'Graphs' tab)")
    except FileNotFoundError:
        console.print(f"[yellow]tensorboard CLI not found on PATH[/yellow] — logs are still written to {out_dir}")


if __name__ == "__main__":
    main()
