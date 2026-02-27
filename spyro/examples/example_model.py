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
            dictionary[key] = deepcopy(default[key])  # For safety to avoid mutation
        elif isinstance(default[key], dict):
            recursive_dictionary_substitution(dictionary[key], default[key])


class ExampleModelBase:
    """Base class for example models with dictionary handling."""
    def __init__(self, dictionary=None, default_dictionary=None, comm=None):
        self.optional_dictionary = deepcopy(dictionary)
        self.default_dictionary = default_dictionary
        if dictionary is None:
            dictionary = {}
        recursive_dictionary_substitution(dictionary, default_dictionary)
        self.input_dictionary = dictionary


class Example_model_acoustic(ExampleModelBase, AcousticWave):
    """Sets up a basic model parameter class for examples and test case models.
    It has the option of reading a dictionary, and if any parameter is missing
    from
    this dictioanry it calls on a default value, that should be defined in the
    relevant
    example file.

    Example Setup

    These examples are intended as reusable velocity model configurations to assist in the development and testing of new methods, such as optimization algorithms, time-marching schemes, or inversion techniques.

    Unlike targeted test cases, these examples do not have a specific objective or expected result. Instead, they provide standardized setups, such as Camembert, rectangular, and Marmousi velocity models, that can be quickly reused when prototyping, testing, or validating new functionality.

    By isolating the setup of common velocity models, we aim to reduce boilerplate and encourage consistency across experiments.

    Feel free to adapt these templates to your needs.

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
        super().__init__(dictionary=dictionary, default_dictionary=default_dictionary, comm=comm)
        AcousticWave.__init__(self, dictionary=self.input_dictionary, comm=comm)


class Example_model_acoustic_FWI(ExampleModelBase, FullWaveformInversion):
    """Sets up a basic model parameter class for examples and test case models.
    It has the option of reading a dictionary, and if any parameter is missing
    from
    this dictioanry it calls on a default value, that should be defined in the
    relevant
    example file.

    Example Setup

    These examples are intended as reusable velocity model configurations to assist in the development and testing of new methods, such as optimization algorithms, time-marching schemes, or inversion techniques.

    Unlike targeted test cases, these examples do not have a specific objective or expected result. Instead, they provide standardized setups, such as Camembert, rectangular, and Marmousi velocity models, that can be quickly reused when prototyping, testing, or validating new functionality.

    By isolating the setup of common velocity models, we aim to reduce boilerplate and encourage consistency across experiments.

    Feel free to adapt these templates to your needs.

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
        super().__init__(dictionary=dictionary, default_dictionary=default_dictionary, comm=comm)
        FullWaveformInversion.__init__(self, dictionary=self.input_dictionary, comm=comm)
