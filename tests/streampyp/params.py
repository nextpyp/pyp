
import logging
import pytest

from pyp.streampyp.params import get_params_file_path, parse_params_from_str, ParamsConfig


# init test logging
logger = logging.getLogger('test')


def test_params_file_path():
    assert get_params_file_path([]) is None
    assert get_params_file_path(['pyp', '-foo=bar']) is None
    assert get_params_file_path(['pyp', '--foo=bar']) is None
    assert get_params_file_path(['pyp', '-bar=bar', '-params_file=path']) is None
    assert get_params_file_path(['pyp', '-params_file=path']) == 'path'
    assert get_params_file_path(['pyp', '--params_file=path']) == 'path'
    assert get_params_file_path(['pyp', '--params_file=/foo bar/cow']) == '/foo bar/cow'


def test_config_types():

    config = ParamsConfig.from_str('''

        [tabs.test.arg_bool]
        type = "bool"

        [tabs.test.arg_int]
        type = "int"

        [tabs.test.arg_float]
        type = "float"

        [tabs.test.arg_float2]
        type = "float2"

        [tabs.test.arg_str]
        type = "str"

        [tabs.test.arg_enum]
        type = "enum"
        enum = { a="A", b="B" }

        [tabs.test.arg_path]
        type = "path"
        
    ''')

    arg = config.arg('test_arg_bool')
    assert arg is not None
    assert arg.type() == 'bool'
    assert arg.check(True) is True
    assert arg.check(False) is False
    with pytest.raises(Exception):
        assert arg.check("T")
    with pytest.raises(Exception):
        assert arg.check(5)

    arg = config.arg('test_arg_int')
    assert arg is not None
    assert arg.type() == 'int'
    assert arg.check(5) == 5
    with pytest.raises(Exception):
        assert arg.check('foo')
    with pytest.raises(Exception):
        assert arg.check(4.2)

    arg = config.arg('test_arg_float')
    assert arg is not None
    assert arg.type() == 'float'
    assert arg.check(4.2) == 4.2
    assert arg.check(5) == 5.0
    with pytest.raises(Exception):
        assert arg.check('foo')

    arg = config.arg('test_arg_float2')
    assert arg is not None
    assert arg.type() == 'float2'
    assert arg.check([4.2, 5.3]) == [4.2, 5.3]
    assert arg.check([5, 7]) == [5.0, 7.0]
    with pytest.raises(Exception):
        assert arg.check('foo')
    with pytest.raises(Exception):
        assert arg.check([])
    with pytest.raises(Exception):
        assert arg.check(['a', 'b'])
    with pytest.raises(Exception):
        assert arg.check([4.2, 'b'])

    arg = config.arg('test_arg_str')
    assert arg is not None
    assert arg.type() == 'str'
    assert arg.check('foo') == 'foo'
    with pytest.raises(Exception):
        assert arg.check(5)

    arg = config.arg('test_arg_enum')
    assert arg is not None
    assert arg.type() == 'enum'
    assert arg.check('a') == 'a'
    assert arg.check('b') == 'b'
    with pytest.raises(Exception):
        assert arg.check('c')
    with pytest.raises(Exception):
        assert arg.check(5)

    arg = config.arg('test_arg_path')
    assert arg is not None
    assert arg.type() == 'path'
    assert arg.check('/the/file') == '/the/file'
    with pytest.raises(Exception):
        assert arg.check(5)


def test_empty():

    config = ParamsConfig.from_str('''
        [tabs]
        # none
    ''')
    params = parse_params_from_str(config, '')

    assert len(params) == 0


    config = ParamsConfig.from_str('''
        [tabs.test.arg]
        type = "int"
    ''')
    params = parse_params_from_str(config, '')

    assert params['test_arg'] is None


def test_types():

    config = ParamsConfig.from_str('''

        [tabs.test.arg_bool]
        type = "bool"

        [tabs.test.arg_int]
        type = "int"

        [tabs.test.arg_float]
        type = "float"

        [tabs.test.arg_float2]
        type = "float2"

        [tabs.test.arg_str]
        type = "str"

        [tabs.test.arg_enum]
        type = "enum"
        enum = { a="A", b="B" }

        [tabs.test.arg_path]
        type = "path"
        
    ''')

    params = parse_params_from_str(config, '''
        test_arg_bool = true
        test_arg_int = 5
        test_arg_float = 4.2
        test_arg_float2 = [5, 7]
        test_arg_str = 'foo'
        test_arg_enum = 'b'
        test_arg_path = '/foo'
    ''')

    assert params['test_arg_bool'] is True
    assert params['test_arg_int'] == 5
    assert params['test_arg_float'] == 4.2
    assert params['test_arg_float2'] == [5.0, 7.0]
    assert params['test_arg_str'] == 'foo'
    assert params['test_arg_enum'] == 'b'
    assert params['test_arg_path'] == '/foo'


def test_defaults():

    config = ParamsConfig.from_str('''

        [tabs.test.arg_no_default]
        type = "int"

        [tabs.test.arg_has_default]
        type = "int"
        default = 5
        
    ''')

    params = parse_params_from_str(config, "")

    assert params['test_arg_no_default'] is None
    assert params['test_arg_has_default'] == 5

    params = parse_params_from_str(config, '''
        test_arg_no_default = 7
        test_arg_has_default = 42
    ''')

    assert params['test_arg_no_default'] == 7
    assert params['test_arg_has_default'] == 42


def test_default_ref():

    config = ParamsConfig.from_str('''
    
        [tabs.test.arg_src]
        type = "int"
        default = 5
        
        [tabs.test.arg_ref]
        type = "int"
        default = { ref="arg_src" }
    ''')

    params = parse_params_from_str(config, '')

    assert params['test_arg_src'] == 5
    assert params['test_arg_ref'] == 5

    params = parse_params_from_str(config, '''
        test_arg_src = 42
    ''')

    assert params['test_arg_src'] == 42
    assert params['test_arg_ref'] == 42

    params = parse_params_from_str(config, '''
        test_arg_src = 42
        test_arg_ref = 7
    ''')

    assert params['test_arg_src'] == 42
    assert params['test_arg_ref'] == 7

    params = parse_params_from_str(config, '''
        test_arg_ref = 7
    ''')

    assert params['test_arg_src'] == 5
    assert params['test_arg_ref'] == 7
