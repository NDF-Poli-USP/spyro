""""Unit tests for the error utilities implemented in spyro.utils.modal.error_management."""
from pytest import fail, raises
from firedrake import Function, FunctionSpace, Mesh
from firedrake.functionspaceimpl import WithGeometry
from numpy import array, inf, nan
from numpy.testing import assert_array_equal
from unittest.mock import Mock, patch
from ufl.form import Form
from ufl.geometry import SpatialCoordinate
from enum import Enum
from spyro.utils.error_management import (
    enum_parameter_error, mutually_exclusive_parameter_error, sanitize_num_array,
    type_data_structure_error, type_firedrake_error, value_file_error,
    value_model_dimension_error, value_numerical_error,
    value_parameter_error, value_string_error)


class ExampleEnum(Enum):
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
    """Tests for sanitize_num_array function."""

    def test_clean_nan_values(self):
        """Test cleaning NaN values."""
        arr = array([1.0, nan, 3.0, nan, 5.0])
        result = sanitize_num_array(arr)
        expected = array([1.0, 0.0, 3.0, 0.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_inf_values(self):
        """Test cleaning inf values."""
        arr = array([1.0, inf, 3.0, -inf, 5.0])
        result = sanitize_num_array(arr)
        expected = array([1.0, 0.0, 3.0, 0.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_negative_values(self):
        """Test cleaning negative values."""
        arr = array([1.0, -2.0, 3.0, -4.0, 5.0])
        result = sanitize_num_array(arr)
        expected = array([1.0, 0.0, 3.0, 0.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_mixed_invalid_values(self):
        """Test cleaning mixed invalid values."""
        arr = array([1.0, nan, -2.0, inf, 5.0, -inf])
        result = sanitize_num_array(arr)
        expected = array([1.0, 0.0, 0.0, 0.0, 5.0, 0.0])
        assert_array_equal(result, expected)

    def test_clean_already_clean_values(self):
        """Test with already clean values."""
        arr = array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sanitize_num_array(arr)
        expected = array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_array_equal(result, expected)

    def test_clean_only_nan(self):
        """Test cleaning only NaN values."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = sanitize_num_array(
            arr, nan_values=True, inf_values=False, negative_values=False)
        expected = array([1.0, 0.0, -2.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_only_inf(self):
        """Test cleaning only infinite values."""
        arr = array([1.0, nan, -2.0, inf, 5.0, -inf])
        result = sanitize_num_array(
            arr, nan_values=False, inf_values=True, negative_values=False)
        expected = array([1.0, nan, -2.0, 0.0, 5.0, 0.0])
        assert_array_equal(result, expected)

    def test_clean_only_negative(self):
        """Test cleaning only negative values."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = sanitize_num_array(
            arr, nan_values=False, inf_values=False, negative_values=True)
        expected = array([1.0, nan, 0.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_nan_and_negative(self):
        """Test cleaning NaN and negative values only."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = sanitize_num_array(
            arr, nan_values=True, inf_values=False, negative_values=True)
        expected = array([1.0, 0.0, 0.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_disabled(self):
        """Test with all cleaning options disabled."""
        arr = array([1.0, nan, -2.0, inf, 5.0])
        result = sanitize_num_array(
            arr, nan_values=False, inf_values=False, negative_values=False)
        expected = array([1.0, nan, -2.0, inf, 5.0])
        assert_array_equal(result, expected)

    def test_clean_type_error(self):
        """Test with invalid input type."""
        with raises(TypeError):
            sanitize_num_array([1, 2, 3])  # list instead of array


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
        result = enum_parameter_error("test_param", ExampleEnum.VALUE1, ExampleEnum)
        assert result == ExampleEnum.VALUE1

    def test_valid_string_value(self):
        """Test with valid string value."""
        result = enum_parameter_error("test_param", "value2", ExampleEnum)
        assert result == ExampleEnum.VALUE2

    def test_invalid_string_value(self):
        """Test with invalid string value."""
        with raises(ValueError) as exc_info:
            enum_parameter_error("test_param", "invalid", ExampleEnum)
        assert "Invalid test_param: 'invalid'" in str(exc_info.value)

    def test_invalid_type(self):
        """Test with invalid type(not enum instance or string)."""
        with raises(TypeError) as exc_info:
            enum_parameter_error("test_param", 123, ExampleEnum)
        assert "'test_param' must be ExampleEnum or str" in str(exc_info.value)


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

    def test_valid_array2D(self):
        """Test with valid 2D numpy array."""
        arr = array([[1, 2], [3, 4]])
        result = type_data_structure_error("test_param", arr, "array2D")
        assert_array_equal(result, arr)

    def test_valid_array3D(self):
        """Test with valid 3D numpy array."""
        arr = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = type_data_structure_error("test_param", arr, "array3D")
        assert_array_equal(result, arr)

    def test_array2D_wrong_dimensions(self):
        """Test with array2D but 3D array provided."""
        arr = array([[[1, 2], [3, 4]]])
        with raises(ValueError) as exc_info:
            type_data_structure_error("test_param", arr, "array2D")
        assert "'test_param' must be a 2D array" in str(exc_info.value)

    def test_array3D_wrong_dimensions(self):
        """Test with array3D but 2D array provided."""
        arr = array([[1, 2], [3, 4]])
        with raises(ValueError) as exc_info:
            type_data_structure_error("test_param", arr, "array3D")
        assert "'test_param' must be a 3D array" in str(exc_info.value)

    def test_shape_check_match(self):
        """Test with matching shape."""
        arr = array([[1, 2], [3, 4]])
        result = type_data_structure_error("test_param", arr, "array2D",
                                           expected_shape=(2, 2))
        assert_array_equal(result, arr)

    def test_shape_check_with_none_dimension(self):
        """Test with shape containing None dimension."""
        arr = array([[1, 2, 3], [4, 5, 6]])
        result = type_data_structure_error("test_param", arr, "array2D",
                                           expected_shape=(2, None))
        assert_array_equal(result, arr)

    def test_shape_check_mismatch(self):
        """Test with shape mismatch."""
        arr = array([[1, 2], [3, 4]])
        with raises(ValueError) as exc_info:
            type_data_structure_error("test_param", arr, "array2D",
                                      expected_shape=(3, 2))
        assert "'test_param' must have shape (3, 2)" in str(exc_info.value)

    def test_element_type_check_for_array(self):
        """Test element type check for numpy array."""
        arr = array([1, 2, 3])
        result = type_data_structure_error("test_param", arr, "array",
                                           expected_type_element="int")
        assert_array_equal(result, arr)

    def test_element_type_check_for_array_failure(self):
        """Test element type check failure for numpy array."""
        arr = array([1, 2.5, 3])
        with raises(TypeError) as exc_info:
            type_data_structure_error("test_param", arr, "array",
                                      expected_type_element="int")
        assert "All elements of 'test_param' must be of type: 'int'" in str(exc_info.value)

    def test_element_type_check_single_string(self):
        """Test with single element type check as string (not tuple)."""
        result = type_data_structure_error("test_param", ["a", "b", "c"], "list",
                                           expected_type_element="str")
        assert result == ["a", "b", "c"]

    def test_invalid_expected_type_element(self):
        """Test with invalid expected_type_element."""
        with raises(ValueError) as exc_info:
            type_data_structure_error("test_param", [1, 2, 3], "list",
                                      expected_type_element="invalid")
        assert "Invalid expected_type_element: 'invalid'" in str(exc_info.value)

    def test_mixed_element_types_valid(self):
        """Test with mixed valid element types."""
        result = type_data_structure_error(
            "test_param", [1, "two", 3.0, None], "list",
            expected_type_element=("int", "str", "float", "NoneType"))
        assert result == [1, "two", 3.0, None]


class TestTypeFiredrakeError:
    """Tests for type_firedrake_error function."""

    @patch('firedrake.Function')
    def test_valid_function(self, mock_function):
        """Test with valid Function."""
        mock_func = Mock(spec=Function)
        result = type_firedrake_error("test_param", mock_func, "Function")
        assert result == mock_func

    def test_none_without_default_for_function(self):
        """Test with None when none_default is False for Function."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", None, "Function")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_with_default_for_function(self):
        """Test with None when none_default is True for Function."""
        result = type_firedrake_error("test_param", None, "Function", none_default=True)
        assert result is None

    def test_function_invalid_type(self):
        """Test with invalid type for Mesh."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "not a function", "Function")
        assert "'test_param' must be of type:" in str(exc_info.value)

    @patch('firedrake.FunctionSpace')
    def test_valid_functionspace(self, mock_fs):
        """Test with valid FunctionSpace."""
        mock_fs_instance = Mock(spec=FunctionSpace)
        result = type_firedrake_error("test_param", mock_fs_instance, "FunctionSpace")
        assert result == mock_fs_instance

    def test_functionspace_with_geometry_instance(self):
        """Test that WithGeometry instances are accepted for FunctionSpace."""
        mock_with_geom = Mock(spec=WithGeometry)
        result = type_firedrake_error("test_param", mock_with_geom, "FunctionSpace")
        assert result == mock_with_geom

    def test_functionspace_with_invalid_type(self):
        """Test with invalid type for FunctionSpace."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "not a functionspace", "FunctionSpace")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_without_default_for_functionspace(self):
        """Test with None when none_default is False for FunctionSpace."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", None, "FunctionSpace")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_with_default_for_functionspace(self):
        """Test with None when none_default is True for FunctionSpace."""
        result = type_firedrake_error(
            "test_param", None, "FunctionSpace", none_default=True)
        assert result is None

    def test_error_message_format_for_functionspace(self):
        """Test that error message correctly shows both FunctionSpace and WithGeometry."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "invalid", "FunctionSpace")
        error_msg = str(exc_info.value)
        assert "'test_param' must be of type:" in error_msg
        assert "FunctionSpace" in error_msg or "WithGeometry" in error_msg

    @patch('firedrake.Mesh')
    def test_valid_mesh(self, mock_mesh):
        """Test with valid Mesh."""
        mock_mesh_instance = Mock(spec=Mesh)
        result = type_firedrake_error("test_param", mock_mesh_instance, "Mesh")
        assert result == mock_mesh_instance

    def test_mesh_invalid_type(self):
        """Test with invalid type for Mesh."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "not a mesh", "Mesh")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_without_default_for_mesh(self):
        """Test with None when none_default is False for Mesh."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", None, "Mesh")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_with_default_for_mesh(self):
        """Test with None when none_default is True for Mesh."""
        result = type_firedrake_error("test_param", None, "Mesh", none_default=True)
        assert result is None

    @patch('ufl.geometry.SpatialCoordinate')
    def test_valid_spatial_coordinate(self, mock_sc_instance):
        """Test with valid SpatialCoordinate."""
        mock_sc_instance = Mock(spec=SpatialCoordinate)
        result = type_firedrake_error("test_param", mock_sc_instance, "SpatialCoordinate")
        assert result == mock_sc_instance

    def test_spatial_coordinate_invalid_type(self):
        """Test with invalid type for SpatialCoordinate."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "not a coordinate", "SpatialCoordinate")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_without_default_for_spatial_coordinate(self):
        """Test with None when none_default is False for SpatialCoordinate."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", None, "SpatialCoordinate")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_with_default_for_spatial_coordinate(self):
        """Test with None when none_default is True for SpatialCoordinate."""
        result = type_firedrake_error("test_param", None, "SpatialCoordinate",
                                      none_default=True)
        assert result is None

    def test_form_type(self):
        """Test with valid Form."""
        mock_form = Mock(spec=Form)
        result = type_firedrake_error("test_param", mock_form, "Form")
        assert result == mock_form

    def test_form_invalid_type(self):
        """Test with invalid type for Form."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", "not a form", "Form")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_without_default_for_form(self):
        """Test with None when none_default is False for Form."""
        with raises(TypeError) as exc_info:
            type_firedrake_error("test_param", None, "Form")
        assert "'test_param' must be of type:" in str(exc_info.value)

    def test_none_with_default_for_form(self):
        """Test with None when none_default is True for Form."""
        result = type_firedrake_error("test_param", None, "Form", none_default=True)
        assert result is None

    def test_invalid_expected_type(self):
        """Test with invalid expected_type."""
        with raises(ValueError) as exc_info:
            type_firedrake_error("test_param", "anything", "Invalid")
        assert "Invalid expected_type: 'Invalid'" in str(exc_info.value)


class TestValueFileError:
    """Tests for type_firedrake_error function."""

    def test_valid_file_extension(self):
        """Test with valid file extension."""
        result = value_file_error("test_param", "file.txt", [".txt", ".csv"])
        assert result == "file.txt"

    def test_valid_file_extension_lowercase(self):
        """Test with lowercase file extension."""
        result = value_file_error("test_param", "file.txt", [".txt", ".csv"])
        assert result == "file.txt"

    def test_valid_file_extension_uppercase(self):
        """Test with uppercase file extension."""
        result = value_file_error("test_param", "file.TXT", [".TXT", ".csv"])
        assert result == "file.TXT"

    def test_valid_file_extension_case_sensitive_fail(self):
        """Test that case-sensitive extension matching fails."""
        with raises(ValueError) as exc_info:
            value_file_error("test_param", "file.TXT", [".txt", ".csv"])
        assert "extension_type" in str(exc_info.value)

    def test_valid_file_extension_mixed_case(self):
        """Test with mixed case file extension."""
        result = value_file_error("test_param", "file.TxT", [".TxT", ".csv"])
        assert result == "file.TxT"

    def test_invalid_file_extension(self):
        """Test with invalid file extension."""
        with raises(ValueError) as exc_info:
            value_file_error("test_param", "file.pdf", [".txt", ".csv"])
        assert "extension_type" in str(exc_info.value)

    def test_none_not_allowed(self):
        """Test with None value when none_default=False."""
        with raises(TypeError) as exc_info:
            value_file_error("test_param", None, [".txt"])
        assert "'test_param' must be a string, got NoneType." in str(exc_info.value)

    def test_none_allowed(self):
        """Test with None value when none_default=True."""
        result = value_file_error("test_param", None, [".txt"], none_default=True)
        assert result is None

    def test_non_string_value(self):
        """Test with non-string value."""
        with raises(TypeError) as exc_info:
            value_file_error("test_param", 123, [".txt"])
        assert "'test_param' must be a string, got int." in str(exc_info.value)

    @patch('os.path.exists')
    def test_file_exists(self, mock_exists):
        """Test with file existence check when file exists."""
        mock_exists.return_value = True
        result = value_file_error("test_param", "file.txt", [".txt"], check_file_existance=True)
        assert result == "file.txt"

    @patch('os.path.exists')
    def test_file_does_not_exist(self, mock_exists):
        """Test with file existence check when file does not exist."""
        mock_exists.return_value = False
        with raises(FileNotFoundError) as exc_info:
            value_file_error("test_param", "missing.txt", [".txt"], check_file_existance=True)
        assert "does not exist" in str(exc_info.value)

    def test_empty_string(self):
        """Test with empty string value."""
        with raises(ValueError) as exc_info:
            value_file_error("test_param", "", [".txt"])
        assert "extension_type" in str(exc_info.value)

    def test_file_without_extension(self):
        """Test with a file name that has no extension."""
        with raises(ValueError) as exc_info:
            value_file_error("test_param", "file", [".txt", ".csv"])
        assert "extension_type" in str(exc_info.value)

    def test_multiple_valid_extensions(self):
        """Test with multiple valid extensions."""
        result = value_file_error("data", "data.csv", [".txt", ".csv", ".json"])
        assert result == "data.csv"
