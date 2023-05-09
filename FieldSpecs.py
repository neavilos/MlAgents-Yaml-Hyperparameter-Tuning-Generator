import math
import random
import sys
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


def _get_or_default(x, key, default):
    return x[key] if key in x else default


class Scale(Enum):
    LINEAR = 1,
    LOG = 2


class InputFieldSpec(ABC):
    def __init__(self, spec_name):
        self.spec_name = spec_name

    @abstractmethod
    def _yield_states(self):
        ...

    @abstractmethod
    def num_states(self):
        ...

    def yield_named_states(self):
        return [(self.spec_name, state) for state in self._yield_states()]


class InputFloatSpec(InputFieldSpec):
    __precision = 7

    def __init__(self, spec_name, values=None, num_samples=1, lower=-math.inf, upper=math.inf, granularity=100,
                 scale=Scale.LINEAR):
        InputFieldSpec.__init__(self, spec_name)
        if values is None:
            values = []
        self.__spec_values = values
        self.__lower = lower
        self.__upper = upper
        self.__num_samples = num_samples
        self.__scale = scale
        self.__sample_granularity = granularity * self.__num_samples

    def __build_sample_space(self):
        match self.__scale:
            case Scale.LINEAR:
                return np.linspace(self.__lower, self.__upper, num=self.__sample_granularity, dtype=float)
            case Scale.LOG:
                return np.logspace(self.__lower, self.__upper, num=self.__sample_granularity, dtype=float)
            case other:
                raise Exception("Invalid scale {0} given!".format(self.__scale))

    def num_states(self):
        return len(self.__spec_values) if self.__spec_values else self.__num_samples

    def _yield_states(self):
        return self.__spec_values if self.__spec_values else [round(float(x), InputFloatSpec.__precision) for x in
                                                              random.sample(list(self.__build_sample_space()),
                                                                            self.__num_samples)]

    @staticmethod
    def create_spec(spec):
        return InputFloatSpec(
            spec_name=spec["name"],
            values=_get_or_default(spec, "values", []),
            num_samples=_get_or_default(spec, "num_samples", 1),
            lower=_get_or_default(spec, "lower", -math.inf),
            upper=_get_or_default(spec, "upper", math.inf),
            granularity=_get_or_default(spec, "granularity", 100)
        )


class InputIntSpec(InputFieldSpec):
    def __init__(self, spec_name, values=[], num_samples=1, lower=0, upper=sys.maxsize, granularity=1,
                 scale=Scale.LINEAR):
        InputFieldSpec.__init__(self, spec_name)
        self.__spec_values = values
        self.__lower = lower
        self.__upper = upper
        self.__num_samples = num_samples
        self.__scale = scale
        self.__sample_granularity = granularity * self.__num_samples

    def __build_sample_space(self):
        match self.__scale:
            case Scale.LINEAR:
                return np.linspace(self.__lower, self.__upper, num=self.__sample_granularity, dtype=int)
            case Scale.LOG:
                return np.logspace(self.__lower, self.__upper, num=self.__sample_granularity, dtype=int)
            case other:
                raise Exception("Invalid scale {0} given!".format(self.__scale))

    def num_states(self):
        return len(self.__spec_values) if self.__spec_values else self.__num_samples

    def _yield_states(self):
        return self.__spec_values if self.__spec_values else [int(x) for x in
                                                              random.sample(list(self.__build_sample_space()),
                                                                            self.__num_samples)]

    @staticmethod
    def create_spec(spec):
        return InputIntSpec(
            spec_name=spec["name"],
            values=_get_or_default(spec, "values", []),
            num_samples=_get_or_default(spec, "num_samples", 1),
            lower=_get_or_default(spec, "lower", -math.inf),
            upper=_get_or_default(spec, "upper", math.inf),
            granularity=_get_or_default(spec, "granularity", 100)
        )


class InputStrSpec(InputFieldSpec):
    def __init__(self, spec_name, spec_values):
        InputFieldSpec.__init__(self, spec_name)
        self.__spec_values = spec_values

    def num_states(self):
        return len(self.__spec_values)

    def _yield_states(self):
        return self.__spec_values

    @staticmethod
    def create_spec(spec):
        return InputStrSpec(
            spec_name=spec["name"],
            spec_values=_get_or_default(spec, "values", [])
        )


class InputBoolSpec(InputFieldSpec):
    def __init__(self, spec_name, is_fixed_state, fixed_state_value=None):
        InputFieldSpec.__init__(self, spec_name)
        self.__is_fixed_state = is_fixed_state
        self.__fixed_state_value = fixed_state_value

    def num_states(self):
        return 1 if self.__is_fixed_state else 2

    def _yield_states(self):
        return [self.__fixed_state_value] if self.__is_fixed_state else [True, False]

    @staticmethod
    def create_spec(spec):
        maybe_value = _get_or_default(spec, "value", None)
        if maybe_value != None:
            return InputBoolSpec(
                spec_name=spec["name"],
                is_fixed_state=True,
                fixed_state_value=maybe_value
            )
        return InputBoolSpec(
            spec_name=spec["name"],
            is_fixed_state=False,
            fixed_state_value=maybe_value
        )


class InputBehaviorSpec(object):
    def __init__(self, behavior_name: str, field_configs: list):
        self.behavior_name = behavior_name
        self.field_configs = field_configs
