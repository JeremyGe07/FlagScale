import copy
import os

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan, summarize_plan
from flagscale.runner.auto_tuner.plan.runtime import build_stage_hetero_runtime_overrides
from flagscale.runner.auto_tuner.plan.summary import (
    is_runtime_executable_plan,
    plan_kind,
    segment_count,
    summarize_execution_contract,
)
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


class Generator:
    _PLAN_STRATEGY_KEYS = (
        "data_parallel_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "micro_batch_size",
        "acc_step",
    )
    _PLAN_METADATA_KEYS = (
        "plan_kind",
        "stage_count",
        "segment_count",
        "runtime_mode",
        "runtime_executable",
        "execution_contract",
        "plan_summary",
    )


    def __init__(self, config):
        self.config = config
        # TODO: Just a temporary solution, need to be configurated by user
        if "args_mapping" in config.experiment.auto_tuner:
            self.args_mapping = config.experiment.auto_tuner.args_mapping
        else:
            self.args_mapping = {
                "data_parallel_size": "data_parallel_size",
                "use_distributed_optimizer": "use_distributed_optimizer",
                "tensor_model_parallel_size": "tensor_model_parallel_size",
                "sequence_parallel": "sequence_parallel",
                "pipeline_model_parallel_size": "pipeline_model_parallel_size",
                "num_layers_per_virtual_pipeline_stage": "num_layers_per_virtual_pipeline_stage",
                "recompute_method": "recompute_method",
                "recompute_granularity": "recompute_granularity",
                "recompute_num_layers": "recompute_num_layers",
                "micro_batch_size": "micro_batch_size",
                "context_parallel_size": "context_parallel_size",
                "expert_model_parallel_size": "expert_model_parallel_size",
                "decoder_first_pipeline_num_layers": "decoder_first_pipeline_num_layers",
                "decoder_last_pipeline_num_layers": "decoder_last_pipeline_num_layers",
            }

    def _set_value(self, strategy, config):
        for key, value in self.args_mapping.items():
            if key in ["micro_batch_size"]:
                config.train.model[value] = strategy[key]
            elif key in ["data_parallel_size"]:
                continue
            else:
                if strategy[key] is None:
                    if value in config.train.system:
                        del config.train.system[value]
                    continue
                config.train.system[value] = strategy[key]

    def _set_auto_tune_train_iters(self, config):
        control = config.experiment.auto_tuner.get("control", {})
        train_iters = control.get("train_iters", 5)
        config.train.model.train_iters = train_iters

        scheduler = config.train.model.optimizer.lr_scheduler
        if "lr_warmup_iters" not in scheduler:
            return

        max_warmup_iters = max(train_iters - 1, 0)
        scheduler.lr_warmup_iters = min(scheduler.lr_warmup_iters, max_warmup_iters)

    def _disable_validation(self, config):
        config.train.model.eval_iters = 0

    def _build_plan_runtime_metadata(self, strategy, config):
        if strategy.get("plan_summary") is not None:
            missing = [key for key in self._PLAN_METADATA_KEYS if strategy.get(key) is None]
            if missing:
                raise ValueError(
                    "strategy plan metadata is incomplete: missing {}".format(", ".join(missing))
                )
            runtime_executable = strategy.get("runtime_executable")
            if not isinstance(runtime_executable, bool):
                raise ValueError("strategy plan metadata runtime_executable must be bool")
            return {
                "plan_kind": strategy.get("plan_kind"),
                "stage_count": strategy.get("stage_count"),
                "segment_count": strategy.get("segment_count"),
                "runtime_mode": strategy.get("runtime_mode"),
                "runtime_executable": runtime_executable,
                "execution_contract": strategy.get("execution_contract"),
                "plan_summary": strategy.get("plan_summary"),
            }
        if not all(key in strategy for key in self._PLAN_STRATEGY_KEYS):
            return None
        plan = lower_strategy_to_plan(strategy, config)
        validation = validate_model_plan(plan)
        return {
            "plan_kind": plan_kind(plan),
            "stage_count": len(plan.stages),
            "segment_count": segment_count(plan),
            "runtime_mode": validation.runtime_mode,
            "runtime_executable": is_runtime_executable_plan(plan),
            "execution_contract": summarize_execution_contract(plan),
            "plan_summary": summarize_plan(plan),
        }

    def _set_plan_metadata(self, strategy, config):
        metadata = self._build_plan_runtime_metadata(strategy, config)
        if metadata is None:
            return
        if not metadata["runtime_executable"]:
            raise ValueError(
                "{} plan is analysis-only and cannot enter executable generator path".format(
                    metadata["plan_kind"]
                )
            )
        config.experiment.auto_tuner.plan = OmegaConf.merge(
            config.experiment.auto_tuner.get("plan", {}),
            {
                "plan_kind": metadata["plan_kind"],
                "stage_count": metadata["stage_count"],
                "segment_count": metadata["segment_count"],
                "runtime_mode": metadata["runtime_mode"],
                "runtime_executable": metadata["runtime_executable"],
                "execution_contract": metadata["execution_contract"],
                "plan_summary": metadata["plan_summary"],
            },
        )
        self._set_stage_hetero_runtime(strategy, config)

    def _set_stage_hetero_runtime(self, strategy, config):
        overrides = build_stage_hetero_runtime_overrides(strategy, config)
        if overrides is None:
            return
        config.train.system.hetero = OmegaConf.merge(
            config.train.system.get("hetero", {}), overrides["hetero"]
        )
        config.train.system = OmegaConf.merge(config.train.system, overrides["system"])

    def gen(self, strategy):
        config = copy.deepcopy(self.config)
        self._set_value(strategy, config)

        # Logging interval should be 1
        config.train.system.logging.log_interval = 1

        # Set redict and tee
        config.experiment.runner.tee = 3
        config.experiment.runner.redirects = 3

        # auto_tune should be true, it will not save ckpt when train ended and report memory every iteration
        config.train.system.auto_tune = True

        # Del lr_warmup_samples and train_samples to run megatron.
        assert "optimizer" in config.train.model
        assert "lr_scheduler" in config.train.model.optimizer
        if "lr_warmup_samples" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_warmup_samples
        # Del lr_decay_samples and train_samples to run megatron.
        if "lr_decay_samples" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_decay_samples
        # Del rampup_batch_size and train_samples to run megatron.
        if "rampup_batch_size" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.rampup_batch_size
        # Del train_samples to run megatron.
        if "train_samples" in config.train.model:
            del config.train.model.train_samples

        # Del checkpoint load
        if "checkpoint" in config.train.system:
            if "load" in config.train.system.checkpoint:
                del config.train.system.checkpoint.load
            if "save_interval" in config.train.system.checkpoint:
                config.train.system.checkpoint.save_interval = 2000

        # Set train_iters of each task and keep the scheduler valid for short autotune runs.
        self._set_auto_tune_train_iters(config)
        self._disable_validation(config)
        self._set_plan_metadata(strategy, config)

        # log dir
        config.experiment.exp_dir = os.path.join(
            config.experiment.exp_dir, "auto_tuner", f"task_{strategy['idx']}"
        )

        return config

    def gen_best_task(self, strategy, config):
        self._set_value(strategy, config)
        self._set_plan_metadata(strategy, config)
        return config


class ServeGenerator(Generator):
    def __init__(self, config):
        self.config = config
        if "args_mapping" in config.experiment.auto_tuner:
            self.args_mapping = config.experiment.auto_tuner.args_mapping
        else:
            self.args_mapping = {
                "tensor_model_parallel_size": "tensor_parallel_size",
                "pipeline_model_parallel_size": "pipeline_parallel_size",
                "instance": "num_replicas",
                "block_size": "block_size",
                "max_num_batched_tokens": "max_num_batched_tokens",
                "max_num_seqs": "max_num_seqs",
                "swap_space": "swap_space",
                "n_gpu_layers": "n_gpu_layers",
                "parallel": "parallel",
                "threads": "threads",
                "chunked_prefill_size": "chunked_prefill_size",
                "max_prefill_tokens": "max_prefill_tokens",
                "page_size": "page_size",
                "max_running_requests": "max_running_requests",
            }

    def _set_value(self, strategy, config):
        serve_config = config.serve
        model_config = None
        for item in serve_config:
            if item.get("serve_id", None) in ("vllm_model", "sglang_model"):
                model_config = item
                break
            else:
                raise ValueError(
                    f"No 'vllm_model' or 'sglang_model' configuration found in task config: {serve_config}"
                )

        if not model_config.get("resources", None):
            model_config["resources"] = {}
        if model_config is None:
            raise ValueError(
                f"No 'vllm_model' or 'sglang_model' configuration found in task config: {serve_config}"
            )

        engine = model_config.engine
        if "engine" in strategy:
            engine = model_config.engine = strategy["engine"]

        for key, value in self.args_mapping.items():
            if key not in strategy:
                continue
            if key == "instance":
                if strategy[key] is None:
                    if value in model_config.resources:
                        del model_config.resources[value]
                    continue
                if value not in model_config.engine_args_specific:
                    model_config.resources = OmegaConf.merge(
                        model_config.resources, {value: strategy[key]}
                    )
                else:
                    model_config.resources[value] = strategy[key]
            else:
                if strategy[key] is None:
                    if value in model_config.engine_args_specific[engine]:
                        del model_config.engine_args_specific[engine][value]
                    continue
                if value not in model_config.engine_args_specific[engine]:
                    model_config.engine_args_specific[engine] = OmegaConf.merge(
                        model_config.engine_args_specific[engine], {value: strategy[key]}
                    )
                else:
                    model_config.engine_args_specific[engine][value] = strategy[key]
        current_tp = model_config.engine_args_specific[engine].get("tensor_parallel_size", 1)
        current_pp = model_config.engine_args_specific[engine].get("pipeline_parallel_size", 1)
        model_config.resources["num_gpus"] = current_tp * current_pp

        if not config.experiment.get("runner", {}).get("deploy", {}).get("use_fs_serve", True):
            del model_config["resources"]

    def gen(self, strategy):
        config = copy.deepcopy(self.config)
        self._set_value(strategy, config)

        # log dir
        config.experiment.exp_dir = os.path.join(
            config.experiment.exp_dir, "auto_tuner", f"task_{strategy['idx']}"
        )

        return config

    def gen_best_task(self, strategy, config):
        self._set_value(strategy, config)
        return config
