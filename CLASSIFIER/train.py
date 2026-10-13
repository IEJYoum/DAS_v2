"""One ordinary training loop; model choice stays in model.build_model."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from CLASSIFIER import data, infer, report
from CLASSIFIER.model import build_model


def cell_weighted_loss(logits, targets, weights):
    total = weights.sum()
    if total.item() <= 0:
        raise ValueError("Training batch has no supervised pixels")
    ce = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
    return (ce * weights).sum() / total


def main(cfg, scenes=None, log_fn=print):
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(cfg["seed"] + 100000)
    scenes = data.load_scenes(cfg["manifest"], cfg) if scenes is None else scenes
    training = [s for s in scenes if s["metadata"]["split"] == "train"]
    validation = [s for s in scenes if s["metadata"]["split"] == "val"]
    if not training or not validation:
        raise ValueError("Training needs separate train and val scenes")
    model = build_model(cfg["model"]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    log_fn(f"Training: {parameter_count:,} parameters, {cfg['steps']} steps, {cfg['device']}, {cfg['threads']} threads")
    best_score = -1.0
    training_seconds = validation_seconds = render_seconds = 0.0
    wall_start = time.perf_counter()
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for step in range(1, cfg["steps"] + 1):
            start = time.perf_counter()
            chosen = rng.integers(len(training), size=cfg["batch_size"])
            augmented = [data.augment_scene(training[i], cfg, rng) for i in chosen]
            arrays = [np.stack([a[i] for a in augmented]) for i in range(4)]
            x = torch.from_numpy(arrays[0]).float().to(cfg["device"])
            targets = torch.from_numpy(arrays[1]).long().to(cfg["device"])
            weights = torch.from_numpy(arrays[2]).to(cfg["device"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = cell_weighted_loss(logits, targets, weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
            training_seconds += time.perf_counter() - start
            event = {"step": step, "loss": loss.item(), "training_seconds": training_seconds}
            if step == 1:
                start = time.perf_counter()
                _, stats = report.write_batch(arrays, out / "debug" / "train_batch.png", cfg["class_names"],
                                              logits.detach().softmax(dim=1).cpu().numpy())
                (out / "debug" / "train_batch.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                render_seconds += time.perf_counter() - start
            if step % cfg["log_every"] == 0 or step == cfg["steps"]:
                event["gradient_norm"] = float(torch.sqrt(sum(p.grad.detach().square().sum()
                                                              for p in model.parameters() if p.grad is not None)).item())
                eta = training_seconds / step * (cfg["steps"] - step)
                log_fn(f"step {step}/{cfg['steps']} | loss {loss.item():.4f} | train {training_seconds:.1f}s | training ETA {eta:.1f}s")
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                start = time.perf_counter()
                predictions, maps = infer.evaluate_scenes(model, validation, cfg)
                scores = report.score_cells(predictions, infer.label_table(validation), cfg["class_names"])
                validation_seconds += time.perf_counter() - start
                event["validation"] = scores
                score = scores["macro_f1"]
                if score > best_score:
                    best_score = score
                    torch.save({"state_dict": model.state_dict(), "config": cfg,
                                "step": step, "validation_macro_f1": score}, out / "best.pt")
                log_fn(f"validation {step} | macro-F1 {score:.3f} | accuracy {scores['accuracy']:.3f} | best {best_score:.3f}")
                start = time.perf_counter()
                scene = validation[0]
                report.write_debug(scene, scene, maps[scene["metadata"]["scene_id"]],
                                   out / "debug" / f"validation_{step:04d}", cfg["class_names"])
                render_seconds += time.perf_counter() - start
            event.update(validation_seconds=validation_seconds, render_seconds=render_seconds)
            event["wall_seconds"] = time.perf_counter() - wall_start
            metrics_file.write(json.dumps(event) + "\n")
            metrics_file.flush()
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                start = time.perf_counter()
                report.write_loss(out / "metrics.jsonl", out / "debug" / "loss_vs_time.png")
                render_seconds += time.perf_counter() - start
    torch.save({"state_dict": model.state_dict(), "config": cfg, "step": cfg["steps"]}, out / "last.pt")
    (out / "training_summary.json").write_text(json.dumps({
        "parameters": parameter_count, "training_seconds": training_seconds,
        "validation_seconds": validation_seconds, "render_seconds": render_seconds,
        "best_validation_macro_f1": best_score}, indent=2), encoding="utf-8")
    return out / "best.pt"
