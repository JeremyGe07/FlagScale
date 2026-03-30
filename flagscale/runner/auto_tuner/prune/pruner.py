from flagscale.runner.auto_tuner.prune.chip import prune_by_chip_profile
from flagscale.runner.auto_tuner.prune.history import _HISTORY_BASED_PRUNE_FUNC
from flagscale.runner.auto_tuner.prune.memory import (
    prune_by_memory_model,
    prune_by_memory_model_util,
)


class Pruner:

    def __init__(self, config):
        self.config = config
        self.pruned_count = 0
        self.pruned_by_memory_model = 0
        self.pruned_by_chip_profile = 0

    def prune(self, strategy, history=None):
        """Prune strategy based on history recorded strategies."""
        if history is None:
            history = []
        not_run = False
        pruned_by_chip, reason = prune_by_chip_profile(self.config, strategy, history)
        if pruned_by_chip:
            self.pruned_by_chip_profile += 1
            self._mark_pruned(strategy, reason)
            not_run = True
        elif "memory_model" in self.config.experiment.auto_tuner:
            if prune_by_memory_model(self.config, strategy, history):
                not_run = True
                self.pruned_by_memory_model += 1
            elif prune_by_memory_model_util(self.config, strategy, history):
                not_run = True
                self.pruned_by_memory_model += 1

        if not not_run:
            for func in _HISTORY_BASED_PRUNE_FUNC:
                if func(self.config, strategy, history):
                    not_run = True
                    break

        history.append(strategy)
        if not_run:
            self.pruned_count += 1
        return not_run

    def _mark_pruned(self, strategy, reason):
        strategy["pruned"] = True
        strategy["performance"] = None
        strategy["pruned_reason"] = reason
