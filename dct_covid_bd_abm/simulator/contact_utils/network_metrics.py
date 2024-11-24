from typing import NamedTuple, Callable


class NxMetric(NamedTuple):
    label: str
    method: Callable
    kwargs: dict = {}
    # Do not assign values to kwargs outside the constructor.
