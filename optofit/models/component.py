
import abc

class Component(object):
    """
    Base class for the components of a model
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._parent = None
        self._children = []
        self._name = 'component'
        self._parameters = []
        self._hyperparameters = []
        self._path = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def children(self):
        return self._children

    @property
    def path(self):
        if self._path is None:
            if self.parent is None:
                self._path = [self.name]
                # return [self.name]
            else:
                self._path = self.parent.path + [self.name]
        return self._path

    @property
    def parameters(self):
        return self._parameters

    @property
    def hyperparameters(self):
        return self._hyperparameters