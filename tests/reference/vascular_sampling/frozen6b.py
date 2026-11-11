"""Exact frozen6b sampling reference, extracted without ML/data execution."""
from collections import Counter
import torch
from torch.utils.data import Sampler

def sampling_weights(self):
        self.support_summary()
        policy = self.config.sampling
        memberships = []
        for rows in self.groups:
            tasks = {task for row in rows for task in self._row_tasks[row["id"]]}
            declared = {str(row["sampling_group"]) for row in rows if "sampling_group" in row}
            if len(declared) > 1:
                raise ValueError("One photo has conflicting sampling_group assignments")
            source = next(iter(declared)) if declared else rows[0]["dataset"]
            if policy == "uniform":
                keys = {("all",)}
            elif policy == "dataset_balanced":
                keys = {(source,)}
            elif policy == "task_balanced":
                keys = {(task,) for task in tasks}
            else:
                keys = {(source, task) for task in tasks}
            memberships.append(keys)
        populations = Counter(key for keys in memberships for key in keys)
        return [sum(1 / populations[key] for key in keys) for keys in memberships]

class CoverageSampler(Sampler):
    """Shuffled complete photo pass followed by source/task-balanced extras."""

    def __init__(self, weights, count, generator, coverage_first=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.count = count
        self.generator = generator
        self.coverage_first = coverage_first

    def __len__(self):
        return self.count

    def __iter__(self):
        first = min(len(self.weights), self.count) if self.coverage_first else 0
        indices = torch.randperm(len(self.weights), generator=self.generator)[:first].tolist()
        if self.count > first:
            indices += torch.multinomial(
                self.weights, self.count - first, replacement=True, generator=self.generator
            ).tolist()
        return iter(indices)
