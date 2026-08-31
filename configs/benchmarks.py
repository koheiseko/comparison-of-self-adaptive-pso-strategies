from dataclasses import dataclass, field
from typing import Any

from opfunu.cec_based.cec2017 import (
    F12017,
    F32017,
    F52017,
    F62017,
    F72017,
    F92017,
    F102017,
    F112017,
    F212017,
)

from src.cec_wrapper import CECFunctionWrapper


def _make_cec_config(
    func_class, ndim: int, report_error: bool = True
) -> dict[str, Any]:
    instance = func_class(ndim=ndim)
    wrapper = CECFunctionWrapper(
        func_instance=instance, report_error=report_error
    )
    lb, ub = wrapper.bounds
    return {
        "function": wrapper,
        "bounds": (lb, ub),
        "f_global": instance.f_global,
        "name": wrapper.name,
    }


CEC2017_SUBSET = {
    "F1_BentCigar": F12017,  # Unimodal — ill-conditioning severo
    "F3_Zakharov": F32017,  # Unimodal — não-separável
    "F5_Rastrigin_SR": F52017,  # Multimodal — versão S+R do seu f2 atual
    "F6_Scaffer": F62017,  # Multimodal — expanded, não-separável
    "F7_LunacekBiRastr": F72017,  # Multimodal — deceptive
    "F9_Levy": F92017,  # Multimodal — flat regions
    "F10_Schwefel": F102017,  # Multimodal — ótimo global distante
    "F11_Hybrid1": F112017,  # Híbrida — subcomponentes aleatórios
    "F21_Composition1": F212017,  # Composição — landscape misto
}


@dataclass
class BenchmarksCEC2017Config:
    ndim: int = 30
    report_error: bool = True
    subset: dict = field(default_factory=lambda: CEC2017_SUBSET)

    def build(self) -> dict[str, Any]:
        return {
            name: _make_cec_config(func_class, self.ndim, self.report_error)
            for name, func_class in self.subset.items()
        }
