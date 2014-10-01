"""
# TODO: Move this to a new module?
Define a few canonical stimuli
"""
import numpy as np

from optofit.utils.utils import get_item_at_path

class StimulusPattern(object):
    """
    Abstract base class for stimuli
    """
    def intensity(self, t):
        """
        Return the stimulus intensity at time t
        """
        return 0


class NoStimulusPattern(StimulusPattern):
    """
    No stimulus (Just extend stimulus)
    """
    def intensity(self, t):
        return np.zeros_like(t)

class GivenStimulusPattern(StimulusPattern):
    """
    No stimulus (Just extend stimulus)
    """
    def __init__(self, t_stim, stim, sample_rate=None):
        self.t_stim = t_stim
        self.stim = stim
        self.sample_rate = sample_rate

    def intensity(self, t):
        if self.sample_rate is not None:
            ind = np.round((t-self.t_stim[0]) * float(self.sample_rate)).astype(np.int)
            return self.stim[ind]
        else:
            # If the stimulus is not given at a fixed frequency
            # then we need to interpolate
            return np.interp(t, self.t_stim, self.stim)

class StepStimulusPattern(StimulusPattern):
    """
    Simple step function stimulus
    """
    def __init__(self, t_on, t_off, v_on):
        self.t_on = t_on
        self.t_off = t_off
        self.v_on = v_on

    def intensity(self, t):
        return (t > self.t_on) * (t < self.t_off) * self.v_on

class PeriodicStepStimulusPattern(StimulusPattern):
    """
    Simple step function stimulus
    """
    def __init__(self, t_on, t_off, on_dur, off_dur, v_on):
        self.t_on = t_on
        self.t_off = t_off
        self.on_dur = on_dur
        self.off_dur = off_dur
        self.v_on = v_on

    def intensity(self, t):
        return (t > self.t_on) * \
               (t < self.t_off) * \
               (np.remainder((t-self.t_on), self.on_dur + self.off_dur) < self.on_dur) * \
               self.v_on


class Stimulus(object):
    """
    Base class for stimulus models. Examples of potential subclasses include:
    - Direct current injections into a compartment
    - Direct fluorescence applied to a single compartment
    - Widefield fluorescence traces (same to all compartments)
    """

    def set_input(self, t, inpt):
        """
        Set the variables in inpt with the appropriate stimulus values
        """
        # TODO: Not sure this is the best place/way to do this
        pass

class DirectCompartmentCurrentInjection(Stimulus):
    """
    Direct current injection into a single compartment
    """
    def __init__(self, compartments, stimuluspattern):
        """

        :param compartments:
        :param stimuluspattern:
        :return:
        """

        if isinstance(compartments, list):
            self.compartments = compartments
        else:
            self.compartments = [compartments]
        self.stimuluspattern = stimuluspattern

    def set_input(self, t, inpt):
        """
        Set the current in each of the connected compartments to the stimulus
        value at time t

        :param t:
        :param inpt:
        :return:
        """
        for c in self.compartments:
            c_inpt = get_item_at_path(inpt, c.path)
            c_inpt['I'] = self.stimuluspattern.intensity(t)

class DirectCompartmentIrradiance(Stimulus):
    """
    Shine laser directly onto the compartment
    """
    def __init__(self, compartments, stimuluspattern):
        """

        :param compartments:
        :param stimuluspattern:
        :return:
        """

        if isinstance(compartments, list):
            self.compartments = compartments
        else:
            self.compartments = [compartments]
        self.stimuluspattern = stimuluspattern

    def set_input(self, t, inpt):
        """
        Set the current in each of the connected compartments to the stimulus
        value at time t

        :param t:
        :param inpt:
        :return:
        """
        for c in self.compartments:
            c_inpt = get_item_at_path(inpt, c.path)
            c_inpt['Irr'] = self.stimuluspattern.intensity(t)
