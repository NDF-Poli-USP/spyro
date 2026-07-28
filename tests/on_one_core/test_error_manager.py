""""Unit tests for the error utilities implemented in spyro.utils.modal.error_management."""
from pytest import fail, raises
from numpy import array, inf, nan
from numpy.testing import assert_array_equal
from enum import Enum
from unittest.mock import Mock, patch
from firedrake import Function, FunctionSpace, Mesh
from ufl.geometry import SpatialCoordinate
from spyro.utils.error_management import (
    clean_inst_num, enum_parameter_error, mutually_exclusive_parameter_error,
    type_data_structure_error, type_firedrake_error, value_model_dimension_error,
    value_numerical_error, value_parameter_error, value_string_error)


class TestEnum(Enum):
    """Test enum class for enum_parameter_error tests."""
    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"


class TestValueParameterError:
    """Tests for value_parameter_error function."""

    def test_valid_parameter(self):
        """Test with valid parameter value."""
        result = value_parameter_error("test_param", "valid", ["valid", "invalid"])
        assert result == "valid"

    def test_invalid_parameter_single(self):
        """Test with invalid parameter when only one valid value."""
        with raises(ValueError) as exc_info:
            value_parameter_error("test_param", "invalid", ["valid"])
        assert "Invalid test_param: 'invalid'" in str(exc_info.value)
        assert "Please use: 'valid'" in str(exc_info.value)

    def test_invalid_parameter_multiple(self):
        """Test with invalid parameter when multiple valid values."""
        with raises(ValueError) as exc_info:
            value_parameter_error("test_param", "invalid", ["valid1", "valid2", "valid3"])
        assert "Invalid test_param: 'invalid'" in str(exc_info.value)
        assert "Please use: 'valid1', 'valid2' or 'valid3'" in str(exc_info.value)

    def test_invalid_parameter_two_values(self):
        """Test with invalid parameter when two valid values."""
        with raises(ValueError) as exc_info:
            value_parameter_error("test_param", "invalid", ["valid1", "valid2"])
        assert "Please use: 'valid1' or 'valid2'" in str(exc_info.value)


class TestMutuallyExclusiveParameterError:
    """Tests for mutually_exclusive_parameter_error function."""

    def test_multiple_parameters_defined(self):
        """Test with multiple parameters defined."""
        par_names = ["param1", "param2", "param3"]
        par_values = ["value1", "value2", None]

        with raises(ValueError) as exc_info:
            mutually_exclusive_parameter_error(par_names, par_values)

        error_msg = str(exc_info.value)
        assert "Parameters 'param1' and 'param2' mutually exclusive" in error_msg
        assert ("Please specify only one of these parameters: "
                "'param1', 'param2' or 'param3'") in error_msg

    def test_all_parameters_none(self):
        """Test with all parameters None (should not raise)."""
        par_names = ["param1", "param2"]
        par_values = [None, None]

        # Should not raise an exception
        try:
            mutually_exclusive_parameter_error(par_names, par_values)
        except ValueError:
            fail("mutually_exclusive_parameter_error raised ValueError unexpectedly")

    def test_single_parameter_defined(self):
        """Test with only one parameter defined(should not raise)."""
        par_names = ["param1", "param2", "param3"]
        par_values = ["value1", None, None]

        # Should not raise an exception
        try:
            mutually_exclusive_parameter_error(par_names, par_values)
        except ValueError:
            fail("mutually_exclusive_parameter_error raised ValueError unexpectedly")

    def test_three_parameters_defined(self):
        """Test with three parameters defined."""
        par_names = ["param1", "param2", "param3"]
        par_values = ["value1", "value2", "value3"]

        with raises(ValueError) as exc_info:
            mutually_exclusive_parameter_error(par_names, par_values)

        error_msg = str(exc_info.value)
        assert "Parameters 'param1', 'param2' and 'param3' mutually exclusive" in error_msg


class TestValueModelDimensionError:
    """Tests for value_model_dimension_error function."""

    def test_matching_dimensions(self):
        """Test with matching dimensions(should not raise)."""
        par_names = ("coord1", "coord2")
        parameters = ([1, 2], [3, 4])

        # Should not raise an exception
        try:
            value_model_dimension_error(par_names, parameters, 2)
        except ValueError:
            fail("value_model_dimension_error raised ValueError unexpectedly")

    def test_mismatching_dimensions(self):
        """Test with mismatching dimensions."""
        par_names = ("coord1", "coord2")
        parameters = ([1, 2], [3, 4, 5])

        with raises(ValueError) as exc_info:
            value_model_dimension_error(par_names, parameters, 2)

        error_msg = str(exc_info.value)
        assert "Mismatch in domain dimensions" in error_msg
        assert "coord1 (2), coord2 (3)" in error_msg
        assert "expected model dimension (2D)" in error_msg

    def test_both_mismatching_dimensions(self):
        """Test with both parameters mismatching dimensions."""
        par_names = ("coord1", "coord2")
        parameters = ([1, 2, 3], [3, 4])

        with raises(ValueError) as exc_info:
            value_model_dimension_error(par_names, parameters, 2)

        error_msg = str(exc_info.value)
        assert "coord1 (3), coord2 (2)" in error_msg


class TestCleanInstNum:
    """Tests for clean_inst_num function."""

    def test_clean_nan_values(self):
        """Test cleaning NaN values."""
        arr = array([1.0, nan, 3.0, nan, 5.0])
        result = clean_inst_num(arr)
        expected = array([1.0, 0.0, 3.0, 0.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_inf_values(self):
        """Test cleaning inf values."""
        arr = array([1.0, inf, 3.0, -inf, 5.0])
        result = clean_inst_num(arr)
        expected = array([1.0, 0.0, 3.0, 0.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_negative_values(self):
        """Test cleaning negative values."""
        arr = array([1.0, -2.0, 3.0, -4.0, 5.0])
        result = clean_inst_num(arr)
        expected = array([1.0, 0.0, 3.0, 0.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_mixed_invalid_values(self):
        """Test cleaning mixed invalid values."""
        arr = array([1.0, nan, -2.0, inf, 5.0, -inf])
        result = clean_inst_num(arr)
        expected = array([1.0, 0.0, 0.0, 0.0, 5.0, 0.0])
        assert_array_equal(result, expected)

    def test_clean_already_clean_values(self):
        """Test with already clean values."""
        arr = array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = clean_inst_num(arr)
        expected = array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_only_nan(self):
        """Test cleaning only NaN values."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = clean_inst_num(
            arr, nan_values=True, inf_values=False, negative_values=False)
        expected = array([1.0, 0.0, -2.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_only_inf(self):
        """Test cleaning only infinite values."""
        arr = array([1.0, nan, -2.0, inf, 5.0, -inf])
        result = clean_inst_num(
            arr, nan_values=False, inf_values=True, negative_values=False)
        expected = array([1.0, nan, -2.0, 0.0, 5.0, 0.0])
        assert_array_equal(result, expected)

    def test_clean_only_negative(self):
        """Test cleaning only negative values."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = clean_inst_num(
            arr, nan_values=False, inf_values=False, negative_values=True)
        expected = array([1.0, nan, 0.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_nan_and_negative(self):
        """Test cleaning NaN and negative values only."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = clean_inst_num(
            arr, nan_values=True, inf_values=False, negative_values=True)
        expected = array([1.0, 0.0, 0.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_disabled(self):
        """Test with all cleaning options disabled."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = clean_inst_num(
            arr, nan_values=False, inf_values=False, negative_values=False)
        expected = array([1.0, nan, -2.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_type_error(self):
        """Test with invalid input type."""
        with raises(TypeError):
            clean_inst_num([1, 2, 3])  # list instead of array


class TestValueNumericalError:
    """Tests for value_numerical_error function."""

    def test_valid_float(self):
        """Test with valid float value."""
        result = value_numerical_error("test_param", 3.14)
        assert result == 3.14

    def test_valid_integer(self):
        """Test with valid integer value."""
        result = value_numerical_error("test_param", 42)
        assert result == 42

    def test_float_and_integer_allowed(self):
        """Test with both float and integer allowed."""
        result_float = value_numerical_error(
            "test_param", 3.14, float_num=True, integer_num=True)
        result_int = value_numerical_error(
            "test_param", 42, float_num=True, integer_num=True)
        assert result_float == 3.14
        assert result_int == 42

    def test_valid_float_as_int(self):
        """Test with float when integer is expected."""
        with raises(TypeError) as exc_info:
            value_numerical_error("test_param", 3.14, float_num=False, integer_num=True)
        assert "'test_param' must be an integer" in str(exc_info.value)

    def test_valid_int_as_float(self):
        """Test with integer when float is expected."""
        with raises(TypeError) as exc_info:
            value_numerical_error("test_param", 42, float_num=True, integer_num=False)
        assert "'test_param' must be a float" in str(exc_info.value)

    def test_valid_none_default(self):
        """Test with None when none_default is True."""
        result = value_numerical_error("test_param", None, none_default=True)
        assert result is None

    def test_valid_none_not_default(self):
        """Test with None when none_default is False."""
        with raises(TypeError) as exc_info:
            value_numerical_error("test_param", None)
        assert "'test_param' must be a float" in str(exc_info.value)

    def test_valid_string_parameter(self):
        """Test with string parameter(should raise TypeError)."""
        with raises(TypeError) as exc_info:
            value_numerical_error("test_param", "not a number")
        assert "'test_param' must be a float" in str(exc_info.value)

    def test_both_bounds_inclusive(self):
        """Test with both bounds inclusive."""
        result = value_numerical_error("test_param", 5, lower_bound=0, upper_bound=10,
                                       include_lower_bound=True, include_upper_bound=True)
        assert result == 5

        with raises(ValueError) as exc_info:
            value_numerical_error("test_param", -1., lower_bound=0, upper_bound=10,
                                  include_lower_bound=True, include_upper_bound=True)
        assert ("'test_param' must be between 0 and 10 "
                "(both bounds inclusive)") in str(exc_info.value)

        with raises(ValueError) as exc_info:
            value_numerical_error("test_param", 11., lower_bound=0, upper_bound=10,
                                  include_lower_bound=True, include_upper_bound=True)
        assert ("'test_param' must be between 0 and 10 "
                "(both bounds inclusive)") in str(exc_info.value)

    def test_lower_bound_inclusive(self):
        """Test with lower bound inclusive."""
        result = value_numerical_error("test_param", 0, lower_bound=0,
                                       include_lower_bound=True)
        assert result == 0

        result = value_numerical_error("test_param", 5., lower_bound=0,
                                       include_lower_bound=True)
        assert result == 5.

        with raises(ValueError) as exc_info:
            value_numerical_error(
                "test_param", -1., lower_bound=0, include_lower_bound=True)
        assert "'test_param' must be greater than or equal to 0" in str(exc_info.value)

    def test_upper_bound_inclusive(self):
        """Test with upper bound inclusive."""
        result = value_numerical_error("test_param", 10., upper_bound=10,
                                       include_upper_bound=True)
        assert result == 10

        result = value_numerical_error("test_param", 9, upper_bound=10,
                                       include_upper_bound=True)
        assert result == 9.

        with raises(ValueError) as exc_info:
            value_numerical_error("test_param", 11, upper_bound=10,
                                  include_upper_bound=True)
        assert "'test_param' must be less than or equal to 10" in str(exc_info.value)

    def test_both_bounds_exclusive(self):
        """Test with both bounds exclusive(at bound)."""
        result = value_numerical_error("test_param", 6, lower_bound=5, upper_bound=10)
        assert result == 6

        with raises(ValueError) as exc_info:
            value_numerical_error("test_param", 5, lower_bound=5, upper_bound=10)
        assert ("'test_param' must be between 5 and 10 "
                "(both bounds exclusive)") in str(exc_info.value)

        with raises(ValueError) as exc_info:
            value_numerical_error("test_param", 10, lower_bound=5, upper_bound=10)
        assert ("'test_param' must be between 5 and 10 "
                "(both bounds exclusive)") in str(exc_info.value)

    def test_lower_bound_exclusive(self):
        """Test with lower bound exclusive."""
        result = value_numerical_error("test_param", 6., lower_bound=5,
                                       include_lower_bound=False)
        assert result == 6.

        with raises(ValueError) as exc_info:
            value_numerical_error(
                "test_param", 5, lower_bound=5, include_lower_bound=False)
        assert "'test_param' must be greater than 5" in str(exc_info.value)

    def test_upper_bound_exclusive(self):
        """Test with upper bound exclusive."""
        with raises(ValueError) as exc_info:
            value_numerical_error(
                "test_param", 10, upper_bound=10, include_upper_bound=False)
        assert "'test_param' must be less than 10" in str(exc_info.value)

    def test_invalid_bounds(self):
        """Test with invalid bounds(lower > upper)."""
        with raises(ValueError) as exc_info:
            value_numerical_error("test_param", 5, lower_bound=10, upper_bound=0)
        assert "Invalid bounds" in str(exc_info.value)


class TestEnumParameterError:
    """Tests for enum_parameter_error function."""

    def test_valid_enum_instance(self):
        """Test with valid enum instance."""
        result = enum_parameter_error("test_param", TestEnum.VALUE1, TestEnum)
        assert result == TestEnum.VALUE1

    def test_valid_string_value(self):
        """Test with valid string value."""
        result = enum_parameter_error("test_param", "value2", TestEnum)
        assert result == TestEnum.VALUE2

    def test_invalid_string_value(self):
        """Test with invalid string value."""
        with raises(ValueError) as exc_info:
            enum_parameter_error("test_param", "invalid", TestEnum)
        assert "Invalid test_param: 'invalid'" in str(exc_info.value)

    def test_invalid_type(self):
        """Test with invalid type(not enum instance or string)."""
        with raises(TypeError) as exc_info:
            enum_parameter_error("test_param", 123, TestEnum)
        assert "'test_param' must be TestEnum or str" in str(exc_info.value)


class TestValueStringError:
    """Tests for value_string_error function."""

    def test_valid_string(self):
        """Test with valid string."""
        result = value_string_error("test_param", "valid_string")
        assert result == "valid_string"

    def test_valid_none_default(self):
        """Test with None when none_default is True."""
        result = value_string_error("test_param", None, none_default=True)
        assert result is None

    def test_valid_none_not_default(self):
        """Test with None when none_default is False."""
        with raises(TypeError) as exc_info:
            value_string_error("test_param", None)
        assert "'test_param' must be a string" in str(exc_info.value)

    def test_invalid_type(self):
        """Test with invalid type."""
        with raises(TypeError) as exc_info:
            value_string_error("test_param", 123)
        assert "'test_param' must be a string" in str(exc_info.value)


class TestTypeDataStructureError:
    """Tests for type_data_structure_error function."""

    def test_valid_list(self):
        """Test with valid list."""
        result = type_data_structure_error("test_param", [1, 2, 3], "list")
        assert result == [1, 2, 3]

    def test_valid_dict(self):
        """Test with valid dict."""
        result = type_data_structure_error("test_param", {"a": 1, "b": 2}, "dict")
        assert result == {"a": 1, "b": 2}

    def test_valid_tuple(self):
        """Test with valid tuple."""
        result = type_data_structure_error("test_param", (1, 2, 3), "tuple")
        assert result == (1, 2, 3)

    def test_valid_array(self):
        """Test with valid numpy array."""
        arr = array([1, 2, 3])
        result = type_data_structure_error("test_param", arr, "array")
        assert_array_equal(result, arr)

    def test_valid_none_default(self):
        """Test with None when none_default is True."""
        result = type_data_structure_error("test_param", None, "list", none_default=True)
        assert result is None

    def test_valid_none_not_default(self):
        """Test with None when none_default is False."""
        with raises(TypeError) as exc_info:
            type_data_structure_error("test_param", None, "list")
        assert "'test_param' must be a list" in str(exc_info.value)

    def test_invalid_type(self):
        """Test with invalid type."""
        with raises(TypeError) as exc_info:
            type_data_structure_error("test_param", "not a list", "list")
        assert "'test_param' must be a list" in str(exc_info.value)

    def test_invalid_expected_type(self):
        """Test with invalid expected_type."""
        with raises(ValueError) as exc_info:
            type_data_structure_error("test_param", [1, 2, 3], "invalid")
        assert "Invalid expected_type: 'invalid'" in str(exc_info.value)

    def test_expected_length_match(self):
        """Test with matching expected length."""
        result = type_data_structure_error("test_param", [1, 2, 3],
                                           "list", expected_length=3)
        assert result == [1, 2, 3]

    def test_expected_length_mismatch(self):
        """Test with mismatching expected length."""
        with raises(ValueError) as exc_info:
            type_data_structure_error("test_param", [1, 2, 3],
                                      "list", expected_length=4)
        assert "'test_param' must have length 4" in str(exc_info.value)

    def test_element_type_check_single(self):
        """Test with single element type check."""
        result = type_data_structure_error("test_param", [1, 2, 3], "list",
                                           expected_type_element="int")
        assert result == [1, 2, 3]

    def test_element_type_check_multiple(self):
        """Test with multiple element type check."""
        result = type_data_structure_error("test_param", [1, 2.5, 3], "list",
                                           expected_type_element=("int", "float"))
        assert result == [1, 2.5, 3]

    def test_element_type_check_failure(self):
        """Test with element type check failure."""
        with raises(TypeError) as exc_info:
            type_data_structure_error("test_param", [1, "string", 3], "list",
                                      expected_type_element="int")
        assert ("All elements of 'test_param' must be "
                "of type: 'int'") in str(exc_info.value)

    def test_element_type_check_with_none(self):
        """Test with element type check including None."""
        result = type_data_structure_error("test_param", [1, None, 3], "list",
                                           expected_type_element=("int", "NoneType"))
        assert result == [1, None, 3]


class TestTypeFiredrakeError:
    """Tests for type_firedrake_error function."""

    @patch('firedrake.Function')
    def test_valid_function(self, mock_function):
        """Test with valid Function."""
        mock_func = Mock(spec=Function)
        result = type_firedrake_error("test_param", mock_func, "Function")
        assert result == mock_func

    @patch('firedrake.FunctionSpace')
    def test_valid_functionspace(self, mock_fs):
        """Test with valid FunctionSpace."""
        mock_fs_instance = Mock(spec=FunctionSpace)
        result = type_firedrake_error("test_param", mock_fs_instance, "FunctionSpace")
        assert result == mock_fs_instance

    @patch('firedrake.functionspaceimpl.WithGeometry')
    def test_functionspace_with_geometry(self, mock_fs):
        """Test with valid FunctionSpace."""
        mock_fs_instance = Mock(spec=FunctionSpace)
        result = type_firedrake_error("test_param", mock_fs_instance, "FunctionSpace")
        assert result == mock_fs_instance

    @patch('firedrake.Mesh')
    def test_valid_mesh(self, mock_mesh):
        """Test with valid Mesh."""
        mock_mesh_instance = Mock(spec=Mesh)
        result = type_firedrake_error("test_param", mock_mesh_instance, "Mesh")
        assert result == mock_mesh_instance

    @patch('ufl.geometry.SpatialCoordinate')
    def test_valid_spatial_coordinate(self, mock_sc_instance):
        """Test with valid SpatialCoordinate."""
        mock_sc_instance = Mock(spec=SpatialCoordinate)
        result = type_firedrake_error("test_param", mock_sc_instance, "SpatialCoordinate")
        assert result == mock_sc_instance

    def test_valid_none_default(self):
        """Test with None when none_default is True."""
        result = type_firedrake_error("test_param", None, "Function", none_default=True)
        assert result is None

    def test_valid_none_not_default(self):
        """Test with None when none_default is False."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", None, "Function")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_invalid_type(self):
        """Test with invalid type."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "not a function", "Function")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_invalid_expected_type(self):
        """Test with invalid expected_type."""
        with raises(ValueError) as exc_info:
            type_firedrake_error("test_param", Mock(), "Invalid")
        assert "Invalid expected_type: 'Invalid'" in str(exc_info.value)
