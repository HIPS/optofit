"""
Base classes for parameters and hyperparameters of the model.
A parameter needs a name, a default value, a getter, and a setter.
It may, optionally have a lower and upper bound
"""
import numpy as np
from optofit.inference.distributions import DeltaFunction

class Parameter(object):
    _name = "Parameter"
    _value = None
    _lb = None
    _ub = None
    def __init__(self, name, initial_value=None, distribution=None, lb=-np.Inf, ub=np.Inf):
        self._name = name
        self._lb = lb
        self._ub = ub

        if initial_value is None and distribution is not None:
            # If no initial value is specified, sample the distribution
            self._distribution = distribution
            self._value = self.distribution.sample(1)
        elif initial_value is not None and distribution is not None:
            # It is ok to specify an initial value and a distribution
            self._value = np.atleast_1d(initial_value)
            self._distribution = distribution
        elif initial_value is not None and distribution is None:
            # Initial value and no distribution -> Delta function
            self._value = np.atleast_1d(initial_value)
            self._distribution = DeltaFunction(initial_value)
        else:
            raise Exception('Initial value and distribution cannot both be none!')

        # TODO: Check that distribution has support [lb, ub]

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        assert v >= self.lower_bound, \
            "Value of '%s' must be >= %.f" % (self.name, self.lower_bound)

        assert v <= self.upper_bound, \
            "Value of '%s' must be <= %.f" % (self.name, self.upper_bound)

        self._value = v

    @property
    def distribution(self):
        return self._distribution

    @property
    def lower_bound(self):
        return self._lb

    @property
    def upper_bound(self):
        return self._ub
