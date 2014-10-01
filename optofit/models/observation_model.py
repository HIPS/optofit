"""
This file will get more filled out as we have more observation models, and ways of constructing them.
"""
from parameters import Parameter

def _make_direct_voltage():
    direct = \
        {
            'type': 'observable',
            'observable_type': 'direct',
            'name': 'voltage'
        }
    return direct, [], []

def _make_linear_fluorescence(scale = .5, offset = 65):
    scale = Parameter('scale', scale)
    offset = Parameter('offset', offset)
    linear = \
        {
            'type': 'observable',
            'observable_type': 'linear_transform',
            'name': 'linear_transform',
            'scale': scale,
            'offset': offset
        }
    return linear, [scale, offset], []
