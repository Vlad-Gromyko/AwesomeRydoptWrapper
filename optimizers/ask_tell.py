from rydopt.protocols import GateSystem
from rydopt.pulses.pulse_ansatz import PulseAnsatz
from rydopt.simulation.fidelity import process_fidelity
from rydopt.types import FixedPulseParams, PulseParams

from typing import Iterable, Iterator, List, Tuple, Union

from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def ask(self) -> List[PulseParams]:
        # Оптимизатор запрашивает расчет функции в данных им точках
        pass

    @abstractmethod
    def tell(self, candidates: List[PulseParams] | Tuple[PulseParams], values: List[float] | Tuple[float]):
        # Сообщаем Оптимизатору результаты в запрошенных точках
        pass
