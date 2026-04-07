from abc import ABC, abstractmethod

from flagscale.runner.auto_tuner.utils import (
    sort_by_memory,
    sort_by_memory_model,
    sort_by_performance,
)


class Algo(ABC):
    def __init__(self, strategies, config):
        self.strategies = strategies
        self.config = config

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def has_done(self):
        pass


class GridAlgo(Algo):

    def __init__(self, strategies, config):
        super().__init__(strategies, config)
        self.idx = 0
        use_profiled_time_cost = self.config.experiment.auto_tuner.algo.get(
            "use_profiled_time_cost", False
        )
        chip_scoring_enabled = self.config.experiment.auto_tuner.algo.get(
            "chip_aware_scoring", False
        )
        has_chip_scores = _strategies_have_chip_scores(self.strategies)
        if use_profiled_time_cost:
            # Profiled time is a stronger signal than heuristic chip score; keep chip/memory only as
            # tie-breakers when time-cost ranking is explicitly enabled.
            self.checkout(mode="time_cost")
        elif chip_scoring_enabled and has_chip_scores:
            self.strategies = sorted(self.strategies, key=sort_by_chip_score, reverse=True)
        elif not chip_scoring_enabled and "memory_model" in self.config.experiment.auto_tuner:
            self.checkout(mode="memory_model")

    def checkout(self, mode):
        if mode == "memory":
            if self.idx > 0 and self.idx < len(self.strategies):
                self.strategies = self.strategies[: self.idx] + sorted(
                    self.strategies[self.idx :], key=sort_by_memory
                )
        elif mode == "memory_model":
            self.strategies = sorted(self.strategies, key=sort_by_memory_model, reverse=True)
        elif mode == "time_cost":
            self.strategies = sorted(self.strategies, key=sort_by_time_cost)
        elif mode == "performance":
            if self.idx > 0 and self.idx < len(self.strategies):
                self.strategies = self.strategies[: self.idx] + sorted(
                    self.strategies[self.idx :], key=sort_by_performance
                )

    def search(self):
        """Return a task iteratively."""
        strategy = None
        if self.idx < len(self.strategies):
            strategy = self.strategies[self.idx]
            self.idx += 1
        return strategy

    def has_done(self):
        """Return True if the task space is empyt."""
        if self.idx >= len(self.strategies):
            return True
        return False


def sort_by_chip_score(strategy):
    chip_score = strategy.get("chip_score", float("-inf"))
    memory_model = strategy.get("memory_model", float("-inf"))
    return (chip_score, memory_model)


def sort_by_time_cost(strategy):
    time_cost = strategy["time_cost"]
    chip_score = strategy.get("chip_score", float("-inf"))
    memory_model = strategy.get("memory_model", float("-inf"))
    return (time_cost, -chip_score, -memory_model)


def _strategies_have_chip_scores(strategies):
    if not strategies:
        return False
    return all("chip_score" in strategy for strategy in strategies)
