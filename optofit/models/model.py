"""
Base class for models. Encapsulates the model description
and getters and setters for the parameters of the model.
Also includes hyperparameters
"""
from parameters import Parameter
from optofit.inference.distributions import *
from optofit.neuron.channels import *
from optofit.simulation.stimulus import Stimulus
from collections import defaultdict
from optofit.population.population import Population
from optofit.neuron.neuron import Neuron
from optofit.neuron.compartment import CalciumCompartment
from optofit.neuron.channels import *
from optofit.observation.observable import NewDirectCompartmentVoltage, LowPassCompartmentVoltage, IndependentObservations, LinearFluorescence


class Model(object):
    """
    Scott's proposal for a new model class. Should contain:
    1. a population of neurons
    2. a (set of) observation model(s).
    3. a set of stimuli
    """
    def __init__(self):
        self.population = None
        self.observation = None
        # self.stimuli = []
        self.data_sequences = []

    def add_population(self, population):
        """
        Add a population of neurons to the model.
        """
        assert self.population is None, "Only supporting one population"
        self.population = population

    def add_observation(self, observation):
        """
        Add a new observation to the model. E.g. a fluorescence observation
        would give a noisy version of the voltage of each neuron compartment.
        """
        assert self.observation is None, "Only supporting one population"
        self.observation = observation

    def add_data(self, name, t, stimuli, observations):
        """
        Add a set of stimuli and observations for a given experiment.

        :param stimuli: A dict of {'stimulus name' : {'input name' : value}}
        :param observations: A dict of {'obs name' : {'output name' : value}}
        """
        # TODO: Check that the given stimuli and observations correspond to
        # TODO: stimulus and observation components in the model
        T = len(t)

        # Initialize the latent, state, and input sequences of the population
        Z = np.zeros((T,), dtype=self.population.latent_dtype)
        I = np.zeros((T,), dtype=self.population.input_dtype)
        S = np.zeros((T,), dtype=self.population.state_dtype)

        # Convert the observations to the observation dtype

        # Set the inputs using stimuli
        if isinstance(stimuli, Stimulus):
            stimuli = [stimuli]
        else:
            assert isinstance(stimuli, list)
        for stim in stimuli:
            stim.set_input(t, I)

        # Create a new DataSequence object
        self.data_sequences.append(DataSequence(name, t, stimuli, observations,
                                                Z, I, S))

    def add_data_sequence(self, data_sequence):
        self.data_sequences.append(data_sequence)

def point_parameter_model(values):
    model = Model()
    population = Population('population', model)
    neuron     = Neuron('neuron', population)
    body       = CalciumCompartment('body', neuron)

    channel_constructor_dict = {
        'leak': LeakChannel,
        'ca3kdr': Ca3KdrChannel,
        'ca3ka': Ca3KaChannel,
        'ca3na': Ca3NaChannel,
        'ca3ca': Ca3CaChannel,
        'ca3kahp': Ca3KahpChannel,
        'ca3kc': Ca3KcChannel,
        'chr2': ChR2Channel
    }

    for ch, d in values.iteritems():
        if ch in channel_constructor_dict:
            channel = channel_constructor_dict[ch](
                ch, body,
                Parameter('g_' + ch, d['g'] ,lb=0.0),
                Parameter('E_' + ch, d['E'] ,lb=0.0)
            )
            body.add_channel(channel)
        else:
            print "Warning: ", ch, " not in dict"

    neuron.add_compartment(body, None)
    population.add_neuron(neuron)
    model.add_population(population)

    return model, body


class DataSequence(object):
    """
    Container for data sequences. Holds the time points, the stimuli, the
    observations, and also the latent state sequence of the model.
    """
    def __init__(self, name, t, stimuli, observations, latent, inpt, states):
        self.name = name
        self.t = t
        self.T = len(t)
        self.stimuli = stimuli
        self.observations = observations
        self.latent = latent
        self.input = inpt
        self.states = states


class OldModel(object):

    _description = {}
    _parameters = []
    _hyperparameters = []

    def __init__(self, desc  , params, hypers):
        self._description = desc
        self._parameters = params
        self._hyperparameters = hypers

    @property
    def description(self):
        return self._description

    # The parameters property refers to all of the model settings
    # that will be inferred, alongside the voltage trace.
    @property
    def parameters(self):
        return self._parameters

    # The hyperparameters property refers to settings that will
    # be fixed for a given run of the inference algorithm.
    @property
    def hyperparameters(self):
        return self._hyperparameters

    def get_parameter(self, name):
        for p in self.parameters:
            if p.name == name:
                return p

    def set_parameter(self, name, value):
        parameter_set = False
        for p in self.parameters:
            if p.name == name:
                p.value = value
                parameter_set = True

        if not parameter_set:

            print 'WARNING: Parameter %s not found!' % name

def merge_models(models):
    """
    This function takes in a list of models and merges them
    """
    desc   = defaultdict(dict)
    params = []
    hypers = []

    for m in models:
        for k, v in m.description.iteritems():
            if isinstance(v, dict):
                desc[k].update(v)
            else:
                if k in desc:
                    print "Warning: Overwrote value ", desc[k], " with ", v
                desc[k] = v
        params += m.parameters
        hypers += m.hyperparameters
    
    desc = dict(desc)
    return OldModel(desc, params, hypers)

def make_single_compartment_model(name, channels, comp_type="compartment"):
    """
    Make a single compartment model with the specified channels.
    NOTE: The hyperparameters are hardcoded in this function!
    """
    parameters = []
    hyperparameters = []
    desc = \
    {
        # Neuron wide properties
        'type' : 'neuron',
        'name' : name,
        'C' : 1,

        # Single compartment with only a leak channel
        'compartment1' :
        {
            'type' : 'compartment',
            'compartment_type' : comp_type,
            'name' : 'compartment1',
        }
    }

    # Make the channels
    for ch in channels:
        if ch == 'leak':
            leak, leak_params, leak_hypers = _make_leak_channel()

            # Add the channel to the description
            desc['compartment1']['leak'] = leak

            # Add the parameters and hyperparameters to our list
            parameters.extend(leak_params)
            hyperparameters.extend(leak_hypers)

        elif ch == 'Na':
            na, na_params, na_hypers = _make_na_channel()

            # Add the channel to the description
            desc['compartment1']['Na'] = na

            # Add the parameters and hyperparameters to our list
            parameters.extend(na_params)
            hyperparameters.extend(na_hypers)

        elif ch == 'Ca3Na':
            na, na_params, na_hypers = _make_ca3na_channel()

            # Add the channel to the description
            desc['compartment1']['Ca3Na'] = na

            # Add the parameters and hyperparameters to our list
            parameters.extend(na_params)
            hyperparameters.extend(na_hypers)

        elif ch == 'Kdr':
            kdr, kdr_params, kdr_hypers = _make_kdr_channel()

            # Add the channel to the description
            desc['compartment1']['Kdr'] = kdr

            # Add the parameters and hyperparameters to our list
            parameters.extend(kdr_params)
            hyperparameters.extend(kdr_hypers)

        elif ch == 'Ca3Kdr':
            kdr, kdr_params, kdr_hypers = _make_ca3kdr_channel()

            # Add the channel to the description
            desc['compartment1']['Ca3Kdr'] = kdr

            # Add the parameters and hyperparameters to our list
            parameters.extend(kdr_params)
            hyperparameters.extend(kdr_hypers)

        elif ch == 'Ca3Ka':
            ka, ka_params, ka_hypers = _make_ca3ka_channel()

            # Add the channel to the description
            desc['compartment1']['Ca3Ka'] = ka

            # Add the parameters and hyperparameters to our list
            parameters.extend(ka_params)
            hyperparameters.extend(ka_hypers)

        elif ch == 'Ca3Ca':
            ca, ca_params, ca_hypers = _make_ca3ca_channel()

            # Add the channel to the description
            desc['compartment1']['Ca3Ca'] = ca

            # Add the parameters and hyperparameters to our list
            parameters.extend(ca_params)
            hyperparameters.extend(ca_hypers)

        elif ch == 'Ca3Kahp':
            kahp, kahp_params, kahp_hypers = _make_ca3kahp_channel()

            # Add the channel to the description
            desc['compartment1']['Ca3Kahp'] = kahp

            # Add the parameters and hyperparameters to our list
            parameters.extend(kahp_params)
            hyperparameters.extend(kahp_hypers)

        elif ch == 'Ca3Kc':
            ca3kc, ca3kc_params, ca3kc_hypers = _make_ca3kc_channel()

            # Add the channel to the description
            desc['compartment1']['Ca3Kc'] = ca3kc

            # Add the parameters and hyperparameters to our list
            parameters.extend(ca3kc_params)
            hyperparameters.extend(ca3kc_hypers)

        elif ch == 'ChR2':
            chr2, chr2_params, chr2_hypers = _make_chr2_channel()
            desc['compartment1']['ChR2'] = chr2
            parameters.extend(chr2_params)
            hyperparameters.extend(chr2_hypers)

    return OldModel(desc, parameters, hyperparameters)

def make_single_compartment_observations(observables):
    mapping = {'DirectVoltage': obs_model._make_direct_voltage}

    desc            = {'compartment1': {}}
    parameters      = []
    hyperparameters = []

    comp = desc['compartment1']
    for obs in observables:
        if obs in mapping:
            d, params, hypers = mapping[obs]()
            comp[obs] = d
            parameters.extend(params)
            hyperparameters.extend(hypers)

    return OldModel(desc, parameters, hyperparameters)

def make_single_compartment_model_with_observations(name, channels, observables, comp_type="calcium"):
    """
    Make a single compartment model with the specified channels and observables
    """
    neuron_model      = make_single_compartment_model(name, channels, comp_type) 
    observation_model = make_single_compartment_observations(observables)
    return merge_models([neuron_model, observation_model])

def _make_leak_channel():
    E_leak = Parameter('E_leak', -60.0)
    # Hard code the gamma distribution over the leak conductance
    a_g_leak = Parameter('a_g_leak', 2.0, lb=1.0)
    b_g_leak = Parameter('b_g_leak', 10.0, lb=0.0)
    g_leak = Parameter('g_leak', 0.2,
                       distribution=GammaDistribution(a_g_leak.value, b_g_leak.value),
                       lb=0.0)
    leak = \
        {
            'type' : 'channel',
            'channel_type' : 'leak',
            'name' : 'leak',
            'E' : E_leak,
            # Gamma distributed leak conductance
            'a_g_leak' : a_g_leak,
            'b_g_leak' : b_g_leak,
            'g' : g_leak
        }

    return leak, [E_leak, g_leak], [a_g_leak, b_g_leak]

def _make_na_channel():
    E_na = Parameter('E_Na', 50.0)
    # Hard code the gamma distribution over the leak conductance
    a_g_na = Parameter('a_g_na', 5.0, lb=1.0)
    b_g_na = Parameter('b_g_na', 0.33, lb=0.0)
    g_na = Parameter('g_na', 15.0,
                       distribution=GammaDistribution(a_g_na.value, b_g_na.value),
                       lb=0.0)
    na = \
        {
            'type' : 'channel',
            'channel_type' : 'sodium',
            'name' : 'Na',
            'E' : E_na,
            # Gamma distributed Na conductance
            'a_g_na' : a_g_na,
            'b_g_na' : b_g_na,
            'g' : g_na
        }

    return na, [E_na, g_na], [a_g_na, b_g_na]

def _make_ca3na_channel():
    E_na = Parameter('E_Na', 50.0)
    # Hard code the gamma distribution over the leak conductance
    a_g_na = Parameter('a_g_na', 5.0, lb=1.0)
    b_g_na = Parameter('b_g_na', 0.33, lb=0.0)
    g_na = Parameter('g_ca3na', 15.0,
                       distribution=GammaDistribution(a_g_na.value, b_g_na.value),
                       lb=0.0)
    na = \
        {
            'type' : 'channel',
            'channel_type' : 'ca3_sodium',
            'name' : 'Ca3Na',
            'E' : E_na,
            # Gamma distributed Na conductance
            'a_g_na' : a_g_na,
            'b_g_na' : b_g_na,
            'g' : g_na
        }

    return na, [E_na, g_na], [a_g_na, b_g_na]


def _make_kdr_channel():
    E_kdr = Parameter('E_K', -77.0)
    # Hard code the gamma distribution over the leak conductance
    a_g_kdr = Parameter('a_g_kdr', 6.0, lb=1.0)
    b_g_kdr = Parameter('b_g_kdr', 1.0, lb=0.0)
    g_kdr = Parameter('g_kdr', 6.0,
                       distribution=GammaDistribution(a_g_kdr.value, b_g_kdr.value),
                       lb=0.0)
    kdr = \
        {
            'type' : 'channel',
            'channel_type' : 'Kdr',
            'name' : 'Kdr',
            'E' : E_kdr,
            # Gamma distributed Kdr conductance
            'a_g_kdr' : a_g_kdr,
            'b_g_kdr' : b_g_kdr,
            'g' : g_kdr
        }

    return kdr, [E_kdr, g_kdr], [a_g_kdr, b_g_kdr]

def _make_ca3kdr_channel():
    E_ca3kdr = Parameter('E_K', -80.0)
    # Hard code the gamma distribution over the leak conductance
    a_g_ca3kdr = Parameter('a_g_ca3kdr', 6.0, lb=1.0)
    b_g_ca3kdr = Parameter('b_g_ca3kdr', 1.0, lb=0.0)
    g_ca3kdr = Parameter('g_ca3kdr', 6.0,
                       distribution=GammaDistribution(a_g_ca3kdr.value, b_g_ca3kdr.value),
                       lb=0.0)
    ca3kdr = \
        {
            'type' : 'channel',
            'channel_type' : 'Ca3Kdr',
            'name' : 'Ca3Kdr',
            'E' : E_ca3kdr,
            # Gamma distributed Kdr conductance
            'a_g_kdr' : a_g_ca3kdr,
            'b_g_kdr' : b_g_ca3kdr,
            'g' : g_ca3kdr
        }

    return ca3kdr, [E_ca3kdr, g_ca3kdr], [a_g_ca3kdr, b_g_ca3kdr]

def _make_ca3ka_channel():
    E_ca3ka = Parameter('E_K', -80.0)
    # Hard code the gamma distribution over the leak conductance
    a_g_ca3ka = Parameter('a_g_ca3ka', 2.0, lb=1.0)
    b_g_ca3ka = Parameter('b_g_ca3ka', 2.0, lb=0.0)
    g_ca3ka = Parameter('g_ca3ka', 1.0,
                       distribution=GammaDistribution(a_g_ca3ka.value, b_g_ca3ka.value),
                       lb=0.0)
    ca3ka = \
        {
            'type' : 'channel',
            'channel_type' : 'Ca3Ka',
            'name' : 'Ca3Ka',
            'E' : E_ca3ka,
            # Gamma distributed Ka conductance
            'a_g_ca3ka' : a_g_ca3ka,
            'b_g_ca3ka' : b_g_ca3ka,
            'g' : g_ca3ka
        }

    return ca3ka, [E_ca3ka, g_ca3ka], [a_g_ca3ka, b_g_ca3ka]

def _make_ca3ca_channel():
    E_ca = Parameter('E_Ca', 80.0)
    # Hard code the gamma distribution over the leak conductance

    # I have no idea what this prior should be
    a_g_ca = Parameter('a_g_ca', 2.0, lb=1.0)
    b_g_ca = Parameter('b_g_ca', 2.0, lb=0.0)
    g_ca = Parameter('g_ca3ca', 1.0,
                       distribution=GammaDistribution(a_g_ca.value, b_g_ca.value),
                       lb=0.0)
    ca = \
        {
            'type' : 'channel',
            'channel_type' : 'ca3_calcium',
            'name' : 'Ca3Ca',
            'E' : E_ca,
            # Gamma distributed Ka conductance
            'a_g_ca' : a_g_ca,
            'b_g_ca' : b_g_ca,
            'g' : g_ca
        }

    return ca, [E_ca, g_ca], [a_g_ca, b_g_ca]

def _make_ca3kahp_channel():
    E_kahp = Parameter('E_Kahp', -80)
    # Hard code the gamma distribution over the leak conductance

    # I have no idea what this prior should be
    a_g_kahp = Parameter('a_g_kahp', 2.0, lb=1.0)
    b_g_kahp = Parameter('b_g_kahp', 2.0, lb=0.0)
    g_kahp = Parameter('g_ca3kahp', 1.0,
                       distribution=GammaDistribution(a_g_kahp.value, b_g_kahp.value),
                       lb=0.0)
    kahp = \
        {
            'type' : 'channel',
            'channel_type' : 'Kahp',
            'name' : 'Kahp',
            'E' : E_kahp,
            # Gamma distributed Ka conductance
            'a_g_kahp' : a_g_kahp,
            'b_g_kahp' : b_g_kahp,
            'g' : g_kahp
        }

    return kahp, [E_kahp, g_kahp], [a_g_kahp, b_g_kahp]

def _make_ca3kc_channel():
    E_ca3kc = Parameter('E_Ca3Kc', -80)
    # Hard code the gamma distribution over the leak conductance

    # I have no idea what this prior should be
    a_g_ca3kc = Parameter('a_g_ca3kc', 2.0, lb=1.0)
    b_g_ca3kc = Parameter('b_g_ca3kc', 2.0, lb=0.0)
    g_ca3kc = Parameter('g_ca3kc', 1.0,
                       distribution=GammaDistribution(a_g_ca3kc.value, b_g_ca3kc.value),
                       lb=0.0)
    ca3kc = \
        {
            'type' : 'channel',
            'channel_type' : 'Ca3Kc',
            'name' : 'Ca3Kc',
            'E' : E_ca3kc,
            # Gamma distributed Ka conductance
            'a_g_ca3kc' : a_g_ca3kc,
            'b_g_ca3kc' : b_g_ca3kc,
            'g' : g_ca3kc
        }

    return ca3kc, [E_ca3kc, g_ca3kc], [a_g_ca3kc, b_g_ca3kc]

def _make_chr2_channel():
    E_chr2 = Parameter('E_ChR2', 0)
    # Hard code the gamma distribution over the leak conductance

    # I have no idea what this prior should be
    a_g_chr2 = Parameter('a_g_chr2', 2.0, lb=1.0)
    b_g_chr2 = Parameter('b_g_chr2', 2.0, lb=0.0)
    g_chr2 = Parameter('g_chr2', 1.0,
                       distribution=GammaDistribution(a_g_chr2.value, b_g_chr2.value),
                       lb=0.0)
    chr2 = \
        {
            'type' : 'channel',
            'channel_type' : 'ChR2',
            'name' : 'ChR2',
            'E' : E_chr2,
            # Gamma distributed Ka conductance
            'a_g_chr2' : a_g_chr2,
            'b_g_chr2' : b_g_chr2,
            'g' : g_chr2
        }

    return chr2, [E_chr2, g_chr2], [a_g_chr2, b_g_chr2]
