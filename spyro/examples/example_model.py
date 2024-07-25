from ..solvers.acoustic_wave import AcousticWave
from ..solvers.inversion import FullWaveformInversion
from copy import deepcopy


def get_list(dictionary):
    list = []
    for key in dictionary.keys():
        list.append(key)

    return list


def recursive_dictionary_substitution(dictionary, default):
    keys = get_list(default)
    for key in keys:
        if key not in dictionary:
            dictionary[key] = default[key]
        elif isinstance(default[key], dict):
            recursive_dictionary_substitution(dictionary[key], default[key])


class Example_model_acoustic(AcousticWave):
    """Sets up a basic model parameter class for examples and test case models.
    It has the option of reading a dictionary, and if any parameter is missing
    from
    this dictioanry it calls on a default value, that should be defined in the
    relevant
    example file.

    Parameters:
    -----------
    dictionary: 'python dictionary' (optional): dictionary with changes to the
    default parameters

    default_dictionary: python 'dictionary': default parameters

    Returns:
    --------
    Example_model
    """

    def __init__(self, dictionary=None, default_dictionary=None, comm=None):
        self.optional_dictionary = deepcopy(dictionary)
        self.default_dictionary = default_dictionary
        if dictionary is None:
            dictionary = {}
        recursive_dictionary_substitution(dictionary, default_dictionary)
        self.input_dictionary = dictionary
        super().__init__(dictionary=dictionary, comm=comm)


class Example_model_acoustic_FWI(FullWaveformInversion):
    """Sets up a basic model parameter class for examples and test case models.
    It has the option of reading a dictionary, and if any parameter is missing
    from
    this dictioanry it calls on a default value, that should be defined in the
    relevant
    example file.

    Parameters:
    -----------
    dictionary: 'python dictionary' (optional): dictionary with changes to the
    default parameters

    default_dictionary: python 'dictionary': default parameters

    Returns:
    --------
    Example_model
    """

    def __init__(self, dictionary=None, default_dictionary=None, comm=None):
        self.optional_dictionary = deepcopy(dictionary)
        self.default_dictionary = default_dictionary
        if dictionary is None:
            dictionary = {}
        recursive_dictionary_substitution(dictionary, default_dictionary)
        self.input_dictionary = dictionary
        super().__init__(dictionary=dictionary, comm=comm)
