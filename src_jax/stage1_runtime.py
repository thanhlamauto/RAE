from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from .config_adapter import build_backend_config_dict, load_repo_config
    from .stage2_runtime import _maybe_raise_backend_dependency_hint, _to_config_dict
    from .vendor import activate_backend
except ImportError:
    from config_adapter import build_backend_config_dict, load_repo_config
    from stage2_runtime import _maybe_raise_backend_dependency_hint, _to_config_dict
    from vendor import activate_backend


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _load_stage1_encoder(args: argparse.Namespace) -> Any:
    config_path = Path(args.config).expanduser().resolve()
    repo_cfg, _ = load_repo_config(args.config, args.set_values)
    backend_cfg_dict = build_backend_config_dict(
        repo_cfg,
        config_path=config_path,
        mode="sample",
        image_size=args.image_size,
        precision=args.precision,
        seed=args.seed,
        exp_name=args.exp_name,
        enable_eval=False,
        require_stage2=False,
    )

    activate_backend(args.backend_dir)

    try:
        from utils import initialize as init_utils
        from utils import wandb_utils as backend_wandb
    except ImportError as exc:
        _maybe_raise_backend_dependency_hint(exc)
        raise

    backend_wandb.initialize = lambda *_args, **_kwargs: None
    backend_cfg = _to_config_dict(backend_cfg_dict)
    return init_utils.instantiate_encoder(backend_cfg)


def list_image_files(input_path: str | Path) -> list[Path]:
    resolved = Path(input_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path not found: {resolved}")
    if resolved.is_file():
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {resolved}")
        return [resolved]

    # KAGGLE FAST SCAN: ImageNet Challenge CSV manifests
    if "imagenet-object-localization-challenge" in str(resolved):
        manifest_name = f"LOC_{resolved.name}_solution.csv"
        # Look for the manifest in standard Kaggle locations relative to Data/CLS-LOC
        # Usually lives in the competition root: /kaggle/input/competitions/imagenet-object-localization-challenge/
        potential_roots = [resolved.parents[3], resolved.parents[4]]
        csv_path = None
        for root in potential_roots:
            if (root / manifest_name).exists():
                csv_path = root / manifest_name
                break
        
        if csv_path:
            print(f"[stage1-runtime] Found Kaggle manifest at {csv_path}. Using fast scan...")
            import pandas as pd
            df = pd.read_csv(csv_path)
            image_paths = []
            
            # Check pattern on first file to avoid million per-file exists checks on slow GCSFuse
            first_id = df.iloc[0]["ImageId"]
            class_id = first_id.split("_")[0]
            if (resolved / class_id / f"{first_id}.JPEG").exists():
                print(f"[stage1-runtime] Detected nested layout for {resolved.name}.")
                for img_id in tqdm(df["ImageId"], desc=f"Mapping {resolved.name}", unit="file"):
                    image_paths.append(resolved / img_id.split("_")[0] / f"{img_id}.JPEG")
            else:
                print(f"[stage1-runtime] Detected flat layout for {resolved.name}.")
                for img_id in tqdm(df["ImageId"], desc=f"Mapping {resolved.name}", unit="file"):
                    image_paths.append(resolved / f"{img_id}.JPEG")
            return sorted(image_paths)

    subdirs = sorted([d for d in resolved.iterdir() if d.is_dir()])
    if subdirs:
        print(f"[stage1-runtime] Found {len(subdirs)} subdirectories under {resolved}. Indexing...")
        image_paths = []
        for i, subdir in enumerate(tqdm(subdirs, desc="Indexing subdirs", unit="class"), 1):
            for path in subdir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(path)
        image_paths.sort()
    else:
        print(f"[stage1-runtime] Listing image files from {resolved}... (this can take time for large folders)")
        image_paths = sorted(
            path
            for path in resolved.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not image_paths:
        raise FileNotFoundError(f"No image files found under: {resolved}")
    return image_paths


def load_image_array(image_path: str | Path, image_size: int | None = None) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if image_size is not None:
            # Standard DiT/SiT center crop resize logic
            w, h = rgb.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            rgb = rgb.crop((left, top, left + s, top + s))
            rgb = rgb.resize((image_size, image_size), Image.BICUBIC)
    return np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0


def _load_batch(batch_paths: Sequence[Path], num_workers: int, image_size: int | None = None) -> np.ndarray:
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            images = list(executor.map(lambda p: load_image_array(p, image_size), batch_paths))
    else:
        images = [load_image_array(path, image_size) for path in batch_paths]
    return np.stack(images, axis=0)


def _iter_batches(items: Sequence[Path], batch_size: int) -> Iterator[list[Path]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start_idx in range(0, len(items), batch_size):
        yield list(items[start_idx : start_idx + batch_size])


def finalize_latent_stats(
    latent_sum: np.ndarray,
    latent_sumsq: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("count must be positive when finalizing latent statistics.")
    mean = latent_sum / float(count)
    var = np.maximum(latent_sumsq / float(count) - np.square(mean), 0.0)
    return mean.astype(np.float32), var.astype(np.float32)


def save_latent_stats(
    output_path: str | Path,
    *,
    mean_hwc: np.ndarray,
    var_hwc: np.ndarray,
    count: int,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": torch.from_numpy(np.transpose(mean_hwc, (2, 0, 1))),
        "var": torch.from_numpy(np.transpose(var_hwc, (2, 0, 1))),
        "count": count,
    }
    torch.save(payload, destination)
    return destination


def run_stage1_reconstruction(args: argparse.Namespace) -> Path:
    import jax.numpy as jnp

    encoder = _load_stage1_encoder(args)
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    arr = load_image_array(image_path)[None, ...]
    latents = encoder.encode(jnp.asarray(arr))
    recon = np.asarray(encoder.decode(latents)[0], dtype=np.uint8)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(recon).save(output_path)
    return output_path


def run_stage1_latent_stats(args: argparse.Namespace) -> Path:
    import jax.numpy as jnp

    encoder = _load_stage1_encoder(args)
    input_root = Path(args.input).expanduser().resolve()
    image_paths = list_image_files(input_root)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images selected from: {input_root}")

    latent_sum: np.ndarray | None = None
    latent_sumsq: np.ndarray | None = None
    processed = 0
    total = len(image_paths)
    pbar = tqdm(total=total, desc="Computing stats", unit="img")
    
    # ImageNet typically uses 256 for this RAE setup
    image_size = args.image_size or 256

    for batch_idx, batch_paths in enumerate(_iter_batches(image_paths, args.batch_size), start=1):
        batch = _load_batch(batch_paths, args.num_workers, image_size=image_size)
        latents = np.asarray(encoder.encode(jnp.asarray(batch)), dtype=np.float32)
        batch_sum = latents.sum(axis=0, dtype=np.float64)
        batch_sumsq = np.square(latents, dtype=np.float64).sum(axis=0, dtype=np.float64)

        if latent_sum is None:
            latent_sum = batch_sum
            latent_sumsq = batch_sumsq
        else:
            latent_sum += batch_sum
            latent_sumsq += batch_sumsq

        processed += latents.shape[0]
        pbar.update(latents.shape[0])

    pbar.close()
    assert latent_sum is not None and latent_sumsq is not None
    mean_hwc, var_hwc = finalize_latent_stats(latent_sum, latent_sumsq, processed)
    return save_latent_stats(args.output, mean_hwc=mean_hwc, var_hwc=var_hwc, count=processed)


def run_stage1_folder_reconstruction(args: argparse.Namespace) -> Path:
    import jax.numpy as jnp

    encoder = _load_stage1_encoder(args)
    input_root = Path(args.input).expanduser().resolve()
    image_paths = list_image_files(input_root)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images selected from: {input_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total = len(image_paths)
    pbar = tqdm(total=total, desc="Reconstructing", unit="img")
    relative_root = input_root if input_root.is_dir() else input_root.parent
    output_ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"

    for batch_idx, batch_paths in enumerate(_iter_batches(image_paths, args.batch_size), start=1):
        batch = _load_batch(batch_paths, args.num_workers)
        latents = encoder.encode(jnp.asarray(batch))
        recon_batch = np.asarray(encoder.decode(latents), dtype=np.uint8)

        for source_path, recon in zip(batch_paths, recon_batch, strict=True):
            relative_path = source_path.relative_to(relative_root)
            destination = output_dir / relative_path.with_suffix(output_ext)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(recon).save(destination)

        processed += len(batch_paths)
        pbar.update(len(batch_paths))

    pbar.close()
    return output_dir
