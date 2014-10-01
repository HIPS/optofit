from pybiophys.models.model import Model
from pybiophys.neuron.neuron import Neuron
from pybiophys.neuron.compartment import CalciumCompartment
from pybiophys.population.population import Population
from pybiophys.test.data_utilities import model_dict_to_model
from pybiophys.observation.observable import IndependentObservations, LowPassCompartmentVoltage
from pybiophys.neuron.channels import LeakChannel, Ca3KdrChannel, Ca3KaChannel, Ca3NaChannel, Ca3CaChannel, Ca3KahpChannel, Ca3KcChannel, ChR2Channel
from pybiophys.models.parameters import Parameter
from pybiophys.inference.observation_likelihood import mean_squared_error
import pickle
import numpy as np



def params_to_model(params):
    model = Model()
    population = Population('population', model)
    neuron     = Neuron('neuron', population)
    body       = CalciumCompartment('body', neuron)

    channel_constructor_dict = {
        u'leak': LeakChannel,
        u'ca3kdr': Ca3KdrChannel,
        u'ca3ka': Ca3KaChannel,
        u'ca3na': Ca3NaChannel,
        u'ca3ca': Ca3CaChannel,
        u'ca3kahp': Ca3KahpChannel,
        u'ca3kc': Ca3KcChannel,
        u'chr2': ChR2Channel
    }

    for ch, g in params.iteritems():
        if ch in channel_constructor_dict:
            channel = channel_constructor_dict[ch](
                ch.encode('ascii', 'ignore'), body,
                Parameter('g_' + ch.encode('ascii', 'ignore'), g.tolist()[0] ,lb=0.0),
            )
            body.add_channel(channel)
        else:
            print "Warning: ", ch, " not in dict"

    neuron.add_compartment(body, None)
    population.add_neuron(neuron)
    model.add_population(population)

    observation = IndependentObservations('observations', model)
    lp_body_voltage = LowPassCompartmentVoltage('lp body voltage', model, body, filterbins=20)
    #import pdb; pdb.set_trace()
    lp_body_voltage.sigma.value = params[u'sigma']
    observation.add_observation(lp_body_voltage)
    model.add_observation(observation)
    return model

def no_noise_model(model):
    d = {}
    for key, data in model['neuron'].iteritems():
        d[key] = data['g']

    d[u'sigma'] = model['observable']['sigma']

    return params_to_model(d)

"""
def main(job_id, params):
    print job_id, params
    _, true_model, _ = pickle.load(open("/home/aaron/git/optofit/data/64seconds/test_1.pk"))
    error = mean_squared_error(model_dict_to_model(true_model), params_to_model(params))
    print params, error

    return error
"""
