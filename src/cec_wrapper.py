from dataclasses import dataclass

import numpy as np


@dataclass
class CECFunctionWrapper:
    func_instance: object
    report_error: bool = (
        False  # se True, retorna f(x) - f*; se False, f(x) bruto
    )

    def __call__(self, x: np.ndarray) -> float:
        raw = self.func_instance.evaluate(x)
        if self.report_error:
            return raw - self.func_instance.f_global
        return raw

    @property
    def bounds(self):
        lb = self.func_instance.lb[0]
        ub = self.func_instance.ub[0]
        return (lb, ub)

    @property
    def f_global(self):
        return self.func_instance.f_global

    @property
    def name(self):
        return type(self.func_instance).__name__
