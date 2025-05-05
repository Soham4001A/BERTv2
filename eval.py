"""Script to evaluate a pretrained model."""

import torch
import hydra


import time
import datetime
import logging
from collections import defaultdict

import cramming
import evaluate
from safetensors.torch import load_file as safe_load


log = logging.getLogger(__name__)


def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    tasks = cramming.prepare_task_dataloaders(tokenizer, cfg.eval, cfg.impl)

    metrics = dict()
    stats = defaultdict(list)
    # Start the clocks now:
    for task_name, task in tasks.items():
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        # Prepare model for finetuning:
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        # ------------------------------------------------------------------
        # Build backend *first*, then load only the encoder weights from the
        # pre‑training checkpoint (ignore MLM‑specific layers).  Afterwards
        # create a brand‑new optimiser / scheduler suitable for fine‑tuning
        # so the learning‑rate never collapses to zero.
        # ------------------------------------------------------------------
        model_engine, _, _, _ = cramming.load_backend(
            model, None, tokenizer, cfg.eval, cfg.impl, setup=setup
        )

        # ---- selective checkpoint loading --------------------------------
        # PyTorch ≥ 2.6 sets `weights_only=True` by default which breaks loading
        # checkpoints that contain additional metadata.  Explicitly request the
        # previous behaviour.
        ckpt_state = None
        load_errors = []
        for attempt in ("torch_weights_only", "safetensors", "torch_legacy"):
            try:
                if attempt == "torch_weights_only":
                    ckpt_state = torch.load(model_file, map_location="cpu", weights_only=True)
                elif attempt == "safetensors":
                    ckpt_state = safe_load(model_file)
                elif attempt == "torch_legacy":
                    ckpt_state = torch.load(model_file, map_location="cpu", weights_only=False, pickle_module=pickle)
                break  # success
            except Exception as e:
                load_errors.append((attempt, repr(e)))
                ckpt_state = None
        if ckpt_state is None:
            raise RuntimeError(
                f"Failed to load checkpoint {model_file}. Tried: {load_errors}"
            )
        # Torch‑style checkpoints wrap weights in dict key 'model'; safetensors is already flat.
        if "model" in ckpt_state and isinstance(ckpt_state["model"], dict):
            ckpt_state = ckpt_state["model"]
        encoder_state = {
            k: v
            for k, v in ckpt_state.items()
            if not k.startswith(("decoder", "latent_to_hidden")) and k in model.state_dict()
        }
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        log.info(
            f"Loaded encoder weights.  Missing: {len(missing)} tensors, "
            f"Ignored: {len(unexpected)} tensors."
        )

        # ---- fine‑tune optimiser & scheduler -----------------------------
        num_training_steps = len(task["trainloader"]) * cfg.eval.epochs
        ft_lr = getattr(cfg.eval, "finetune_lr", 2e-5)          # default 2e‑5
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=ft_lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )

        model_engine.optimizer = optimizer
        model_engine.scheduler = scheduler

        try:
            assert task_name != "record"
            metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
        except (FileNotFoundError, AssertionError):  # no specific metric downloadable from evaluate, construct directly
            targets = [evaluate.load(metric_name, cache_dir=cfg.impl.path) for metric_name in task["details"]["target_metrics"]]
            metric = evaluate.CombinedEvaluations(targets)
        # Launch training
        model_engine.train(cfg.eval.eval_in_train_mode)
        loss_vals = []
        for epoch in range(cfg.eval.epochs):
            train_time = time.time()

            for step, batch in enumerate(task["trainloader"]):
                # Heavy lifting is moved to engines
                device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
                loss = model_engine.step(device_batch)
                loss_vals.append(loss.detach())
                if cfg.dryrun:
                    break

            metrics[task_name] = validate(model_engine, task["validloader"], metric, setup, cfg)
            stats[f"{task_name}_epoch"] += [epoch]
            stats[f"{task_name}_loss"] += [loss.item()]

            stats[f"{task_name}_avg_loss"] += [torch.stack(loss_vals).mean().item()]  # Smoothed loss
            loss_vals = []
            current_lr = model_engine.optimizer.param_groups[0]["lr"]

            log_msg = f"Train loss {loss.item():2.4f} at step {step} with lr {current_lr:.5f}. "
            log_msg += f"[Avg: {stats[f'{task_name}_avg_loss'][-1]:2.4f}] after epoch {epoch}."

            stats[f"{task_name}_train_time"] += [(time.time() - train_time)]
            estimated_train_finish = str(datetime.timedelta(seconds=stats[f"{task_name}_train_time"][-1] * cfg.eval.epochs))
            tokens_per_second = (step + 1) * cfg.eval.max_seq_length * cfg.impl.microbatch_size / stats[f"{task_name}_train_time"][-1]
            log_msg += (
                f" Perf: {stats[f'{task_name}_train_time'][-1]/60:2.4f}min per epoch ({tokens_per_second:.0f}t/s). "
                f"Estimated Total Train: {estimated_train_finish}."
            )

            for name, metric_val in metrics[task_name].items():
                stats[f"{task_name}_{name}"] += [metric_val]
            log.info(log_msg)
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in metrics[task_name].items()])
            log.info(f"Validation metric is {msg_metrics} after epoch {epoch}.")
            cramming.utils.wandb_log(stats, cfg)

            if cfg.dryrun:
                break
        # Launch extra testing if extra validation set exists (as with MNLI-mismatched):
        if task["extra_validloader"] is not None:
            extra_eval_metric = validate(model_engine, task["extra_validloader"], metric, setup, cfg)
            # metrics[task_name + "extra"] = extra_eval_metric
            metrics[task_name].update({f"{k}_extra": v for k, v in extra_eval_metric.items()})
            for name, metric_val in extra_eval_metric.items():
                stats[f"{task_name}_{name}_extra"] += [metric_val]
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in extra_eval_metric.items()])
            log.info(f"Extra validation metric is {msg_metrics} after finetuning.")
            cramming.utils.wandb_log({f"{task_name}_{k}_extra": [v] for k, v in extra_eval_metric.items()}, cfg)

    # Check average metric over all tasks:
    target_metrics = []
    for task_name, task in tasks.items():
        target_metric_names = task["details"]["target_metrics"]
        for metric_name in target_metric_names:
            target_metrics.append(metrics[task_name][metric_name])
    metrics[f"{cfg.eval.name}_amean"] = torch.as_tensor(target_metrics).mean().item()
    metrics[f"{cfg.eval.name}_hmean"] = torch.as_tensor(target_metrics).pow(-1).mean().pow(-1).item()
    log.info(f"Overall average metric on evaluation {cfg.eval.name} is {metrics[f'{cfg.eval.name}_amean']:.2f}.")
    cramming.utils.wandb_log(
        {f"{cfg.eval.name}_amean": [metrics[f"{cfg.eval.name}_amean"]], f"{cfg.eval.name}_hmean": [metrics[f"{cfg.eval.name}_hmean"]]},
        cfg,
    )

    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return metrics  # will be dumped into yaml


@torch.no_grad()
def validate(model_engine, validloader, metric, setup, cfg):
    """Evaluate on validation set."""
    model_engine.eval()
    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
        _, predictions = model_engine.forward_inference(**device_batch)

        if getattr(metric, "config_name", "") != "multirc":
            metric.add_batch(predictions=predictions, references=device_batch["labels"])
        else:  # uuuuuughhhhh, whhyyy multirc
            pred_indices = range(step * predictions.shape[0], (step + 1) * predictions.shape[0])
            packages = [dict(idx=validloader.index_lookup[pred_indices[i]], prediction=p) for i, p in enumerate(predictions.cpu())]
            metric.add_batch(predictions=packages, references=batch["labels"])

        if cfg.dryrun and step > 1:
            break

    try:
        eval_metric = metric.compute()
    except ValueError:  # pearson corr computation will raise errors if metric values are NaN
        log.info("Value Error in metrics computation, maybe non-finite values in prediction. Returning backup score.")
        eval_metric = metric.compute(predictions=[0, 1], references=[1, 0])  # spoof terrible result if metric computation fails
    model_engine.train(cfg.eval.eval_in_train_mode)
    return {k: float(v) for k, v in eval_metric.items()}  # force float returns


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")


if __name__ == "__main__":
    launch()
