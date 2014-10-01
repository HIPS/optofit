import numpy as np
from pybiophys.simulation.simulate import simulate
#from pybiophys.models.model        import point_parameter_model
from pybiophys.test.data_utilities import model_dict_to_model

def mean_squared_error(true_model, compare_model):
    #model = model_dict_to_model(compare_model)
    ds = true_model.data_sequences[0]
    return compare_model_to_observations(compare_model, ds.observations, ds.t, ds.input)

def compare_model_to_observations(model, observations, t, stimulus):
    ds = simulate(model, t, stimulus, have_noise=False, I=stimulus)
    return np.mean(((ds.observations.view(np.float64) - observations.view(np.float64)) ** 2).view(np.float64))
