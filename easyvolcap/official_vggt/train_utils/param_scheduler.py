import math


class LinearParamScheduler:
    def __init__(self, start_value: float, end_value: float):
        self.start_value = float(start_value)
        self.end_value = float(end_value)

    def __call__(self, where: float) -> float:
        where = min(max(where, 0.0), 1.0)
        return self.start_value + (self.end_value - self.start_value) * where


class CosineParamScheduler:
    def __init__(self, start_value: float, end_value: float):
        self.start_value = float(start_value)
        self.end_value = float(end_value)

    def __call__(self, where: float) -> float:
        where = min(max(where, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * where))
        return self.end_value + (self.start_value - self.end_value) * cosine


class ConstantParamScheduler:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, where: float) -> float:
        return self.value


class CompositeParamScheduler:
    def __init__(self, schedulers, lengths, interval_scaling=None):
        if len(schedulers) != len(lengths):
            raise ValueError("schedulers and lengths must have the same length")
        self.schedulers = schedulers
        total = float(sum(lengths))
        self.lengths = [float(l) / total if total > 0 else 0.0 for l in lengths]
        if interval_scaling is None:
            interval_scaling = ["rescaled"] * len(self.lengths)
        if len(interval_scaling) != len(self.lengths):
            raise ValueError("interval_scaling and lengths must have the same length")
        self.interval_scaling = interval_scaling

    def __call__(self, where: float) -> float:
        where = min(max(where, 0.0), 1.0)
        start = 0.0
        for idx, length in enumerate(self.lengths):
            end = start + length
            if where <= end or idx == len(self.lengths) - 1:
                local = 0.0
                if length > 0:
                    local = (where - start) / length
                if self.interval_scaling[idx] == "rescaled":
                    return self.schedulers[idx](local)
                return self.schedulers[idx](where)
            start = end
        return self.schedulers[-1](1.0)


def build_param_scheduler(cfg: dict):
    sched_type = cfg.get("type")
    if sched_type is None:
        raise ValueError("param scheduler config missing type")

    if sched_type == "LinearParamScheduler":
        return LinearParamScheduler(cfg["start_value"], cfg["end_value"])
    if sched_type == "CosineParamScheduler":
        return CosineParamScheduler(cfg["start_value"], cfg["end_value"])
    if sched_type == "ConstantParamScheduler":
        return ConstantParamScheduler(cfg["value"])
    if sched_type == "CompositeParamScheduler":
        schedulers = [build_param_scheduler(s) for s in cfg["schedulers"]]
        lengths = cfg["lengths"]
        interval_scaling = cfg.get("interval_scaling")
        return CompositeParamScheduler(schedulers, lengths, interval_scaling)

    raise ValueError(f"Unknown param scheduler type: {sched_type}")
