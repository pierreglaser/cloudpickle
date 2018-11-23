from __future__ import division

import abc
import collections
import base64
import functools
from io import BytesIO
import itertools
import logging
import math
from operator import itemgetter, attrgetter
import pickle
import platform
import random
import subprocess
import sys
import textwrap
import types
import unittest
import weakref
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import pytest

import cloudpickle

from .testutils import subprocess_pickle_echo
from .testutils import assert_run_python_script


_TEST_GLOBAL_VARIABLE = "default_value"


class RaiserOnPickle(object):

    def __init__(self, exc):
        self.exc = exc

    def __reduce__(self):
        raise self.exc


def pickle_depickle(obj, protocol=cloudpickle.DEFAULT_PROTOCOL):
    """Helper function to test whether object pickled with cloudpickle can be
    depickled with pickle
    """
    return pickle.loads(cloudpickle.dumps(obj, protocol=protocol))


class CloudPicklerTest(unittest.TestCase):
    def setUp(self):
        self.file_obj = StringIO()
        self.cloudpickler = cloudpickle.CloudPickler(self.file_obj, 2)


class CloudPickleTest(unittest.TestCase):

    protocol = cloudpickle.DEFAULT_PROTOCOL

    def test_itemgetter(self):
        d = range(10)
        getter = itemgetter(1)

        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

        getter = itemgetter(0, 3)
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

    def test_attrgetter(self):
        class C(object):
            def __getattr__(self, item):
                return item
        d = C()
        getter = attrgetter("a")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("a", "b")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

        d.e = C()
        getter = attrgetter("e.a")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("e.a", "e.b")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

    def test_func_globals(self):
        class Unpicklable(object):
            def __reduce__(self):
                raise Exception("not picklable")

        global exit
        exit = Unpicklable()

        self.assertRaises(Exception, lambda: cloudpickle.dumps(exit))

        def foo():
            sys.exit(0)

        func_code = getattr(foo, '__code__', None)
        if func_code is None:  # PY2 backwards compatibility
            func_code = foo.func_code

        self.assertTrue("exit" in func_code.co_names)
        cloudpickle.dumps(foo)

    def test_buffer(self):
        try:
            buffer_obj = buffer("Hello")
            buffer_clone = pickle_depickle(buffer_obj, protocol=self.protocol)
            self.assertEqual(buffer_clone, str(buffer_obj))
            buffer_obj = buffer("Hello", 2, 3)
            buffer_clone = pickle_depickle(buffer_obj, protocol=self.protocol)
            self.assertEqual(buffer_clone, str(buffer_obj))
        except NameError:  # Python 3 does no longer support buffers
            pass

    def test_lambda(self):
        self.assertEqual(pickle_depickle(lambda: 1)(), 1)

    def test_nested_lambdas(self):
        script = '''
        from tests.cloudpickle_test import pickle_depickle


        a, b = 1, 2
        f1 = lambda x: x + a
        f2 = lambda x: f1(x) // b
        assert pickle_depickle(f2, protocol={protocol})(1) == 1
        '''.format(protocol=self.protocol)

        assert_run_python_script(textwrap.dedent(script))

    def test_closure_none_is_preserved(self):
        def f():
            """a function with no closure cells
            """

        self.assertTrue(
            f.__closure__ is None,
            msg='f actually has closure cells!',
        )

        g = pickle_depickle(f, protocol=self.protocol)

        self.assertTrue(
            g.__closure__ is None,
            msg='g now has closure cells even though f does not',
        )

    def test_dynamically_generated_class_that_uses_super(self):

        script = '''
        from tests.cloudpickle_test import pickle_depickle


        class Base(object):
            def method(self):
                return 1

        class Derived(Base):
            "Derived Docstring"
            def method(self):
                return super(Derived, self).method() + 1

        assert Derived().method() == 2

        # Pickle and unpickle the class.
        UnpickledDerived = pickle_depickle(Derived, protocol={protocol})
        assert UnpickledDerived().method() == 2

        # We have special logic for handling __doc__ because it's a readonly
        # attribute on PyPy.
        assert UnpickledDerived.__doc__ == "Derived Docstring"

        # Pickle and unpickle an instance.
        orig_d = Derived()
        d = pickle_depickle(orig_d, protocol={protocol})
        assert d.method() == 2
        '''.format(protocol=self.protocol)
        assert_run_python_script(textwrap.dedent(script))

    def test_partial(self):
        partial_obj = functools.partial(min, 1)
        partial_clone = pickle_depickle(partial_obj, protocol=self.protocol)
        self.assertEqual(partial_clone(4), 1)

    def test_loads_namespace(self):
        obj = 1, 2, 3, 4
        returned_obj = cloudpickle.loads(cloudpickle.dumps(obj))
        self.assertEqual(obj, returned_obj)

    def test_load_namespace(self):
        obj = 1, 2, 3, 4
        bio = BytesIO()
        cloudpickle.dump(obj, bio)
        bio.seek(0)
        returned_obj = cloudpickle.load(bio)
        self.assertEqual(obj, returned_obj)

    def test_generator(self):

        def some_generator(cnt):
            for i in range(cnt):
                yield i

        gen2 = pickle_depickle(some_generator, protocol=self.protocol)

        assert type(gen2(3)) == type(some_generator(3))
        assert list(gen2(3)) == list(range(3))

    def test_method_descriptors(self):
        f = pickle_depickle(str.upper)
        self.assertEqual(f('abc'), 'ABC')

    def test_instancemethods_without_self(self):
        class F(object):
            def f(self, x):
                return x + 1

        g = pickle_depickle(F.f)
        self.assertEqual(g.__name__, F.f.__name__)
        if sys.version_info[0] < 3:
            self.assertEqual(g.im_class.__name__, F.f.im_class.__name__)
        # self.assertEqual(g(F(), 1), 2)  # still fails

    def test_module(self):
        pickle_clone = pickle_depickle(pickle, protocol=self.protocol)
        self.assertEqual(pickle, pickle_clone)

    def test_module_locals_behavior(self):
        # Makes sure that a local function defined in another module is
        # correctly serialized. This notably checks that the globals are
        # accessible and that there is no issue with the builtins (see #211)

        pickled_func_path = 'local_func_g.pkl'

        child_process_script = '''
        import pickle
        import gc
        with open("{pickled_func_path}", 'rb') as f:
            func = pickle.load(f)

        assert func(range(10)) == 45
        '''

        child_process_script = child_process_script.format(
                pickled_func_path=pickled_func_path)

        try:

            from .testutils import make_local_function

            g = make_local_function()
            with open(pickled_func_path, 'wb') as f:
                cloudpickle.dump(g, f)

            assert_run_python_script(textwrap.dedent(child_process_script))

        finally:
            os.unlink(pickled_func_path)

    def test_correct_globals_import(self):
        script = '''
        import cloudpickle
        import math

        def nested_function(x):
            return x + 1

        def unwanted_function(x):
            return math.exp(x)

        def my_small_function(x, y):
            return nested_function(x) + y

        b = cloudpickle.dumps(my_small_function)

        # Make sure that the pickle byte string only includes the definition
        # of my_small_function and its dependency nested_function while
        # extra functions and modules such as unwanted_function and the math
        # module are not included so as to keep the pickle payload as
        # lightweight as possible.

        assert b'my_small_function' in b
        assert b'nested_function' in b

        assert b'unwanted_function' not in b
        assert b'math' not in b
        '''
        assert_run_python_script(textwrap.dedent(script))

    def test_NoneType(self):
        res = pickle_depickle(type(None), protocol=self.protocol)
        self.assertEqual(type(None), res)


    def test_extended_arg(self):
        # Functions with more than 65535 global vars prefix some global
        # variable references with the EXTENDED_ARG opcode.
        nvars = 65537 + 258
        names = ['g%d' % i for i in range(1, nvars)]
        r = random.Random(42)
        d = {name: r.randrange(100) for name in names}
        # def f(x):
        #     x = g1, g2, ...
        #     return zlib.crc32(bytes(bytearray(x)))
        code = """
        import zlib

        def f():
            x = {tup}
            return zlib.crc32(bytes(bytearray(x)))
        """.format(tup=', '.join(names))
        exec(textwrap.dedent(code), d, d)
        f = d['f']
        res = f()
        data = cloudpickle.dumps([f, f])
        d = f = None
        f2, f3 = pickle.loads(data)
        self.assertTrue(f2 is f3)
        self.assertEqual(f2(), res)

    def test_submodule(self):
        # Function that refers (by attribute) to a sub-module of a package.

        # Choose any module NOT imported by __init__ of its parent package
        # examples in standard library include:
        # - http.cookies, unittest.mock, curses.textpad, xml.etree.ElementTree

        global xml # imitate performing this import at top of file
        import xml.etree.ElementTree
        def example():
            x = xml.etree.ElementTree.Comment # potential AttributeError

        s = cloudpickle.dumps(example)

        # refresh the environment, i.e., unimport the dependency
        del xml
        for item in list(sys.modules):
            if item.split('.')[0] == 'xml':
                del sys.modules[item]

        # deserialise
        f = pickle.loads(s)
        f() # perform test for error

    def test_multiprocess(self):
        func_filename = 'pickled_function.pk'
        grandchild_process_script = """'''
        import pickle
        import textwrap

        with open('{func_filename}', 'rb') as f:
            func = pickle.load(f)
        func()
        '''""".format(func_filename=func_filename)

        child_process_script = '''
        import cloudpickle
        import textwrap

        from testutils import assert_run_python_script


        def example():
            x = xml.etree.ElementTree.Comment


        global xml
        import xml.etree.ElementTree

        with open('{func_filename}', 'wb') as f:
            s = cloudpickle.dump(example, f)

        assert_run_python_script(textwrap.dedent({grandchild_process_script}))
        '''.format(
                grandchild_process_script=grandchild_process_script,
                func_filename=func_filename)

        try:
            assert_run_python_script(textwrap.dedent(child_process_script))
        finally:
            os.unlink(func_filename)

    def test_import(self):
        # like test_multiprocess except subpackage modules referenced directly
        # (unlike test_submodule)
        func_filename = 'pickled_function.pk'

        grandchild_process_script = """'''
        import pickle
        import textwrap

        with open('{func_filename}', 'rb') as f:
            func = pickle.load(f)
        func()
        '''""".format(func_filename=func_filename)

        child_process_script = '''
        import cloudpickle
        import textwrap

        from testutils import assert_run_python_script

        global etree

        def scope():
            def example():
                x = etree.Comment
            return example
        example = scope()
        import xml.etree.ElementTree as etree


        with open('{func_filename}', 'wb') as f:
            s = cloudpickle.dump(example, f)

        assert_run_python_script(textwrap.dedent({grandchild_process_script}))
        '''.format(
                grandchild_process_script=grandchild_process_script,
                func_filename=func_filename)

        try:
            assert_run_python_script(textwrap.dedent(child_process_script))
        finally:
            os.unlink(func_filename)

    def check_logger(self, name):
        logger = logging.getLogger(name)
        pickled = pickle_depickle(logger, protocol=self.protocol)
        self.assertTrue(pickled is logger, (pickled, logger))

        dumped = cloudpickle.dumps(logger)

        code = """if 1:
            import cloudpickle, logging

            logging.basicConfig(level=logging.INFO)
            logger = cloudpickle.loads(%(dumped)r)
            logger.info('hello')
            """ % locals()
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        self.assertEqual(proc.wait(), 0)
        self.assertEqual(out.strip().decode(),
                         'INFO:{}:hello'.format(logger.name))

    def test_logger(self):
        # logging.RootLogger object
        self.check_logger(None)
        # logging.Logger object
        self.check_logger('cloudpickle.dummy_test_logger')

    def test_weakset_identity_preservation(self):
        # Test that weaksets don't lose all their inhabitants if they're
        # pickled in a larger data structure that includes other references to
        # their inhabitants.

        class SomeClass(object):
            def __init__(self, x):
                self.x = x

        obj1, obj2, obj3 = SomeClass(1), SomeClass(2), SomeClass(3)

        things = [weakref.WeakSet([obj1, obj2]), obj1, obj2, obj3]
        result = pickle_depickle(things, protocol=self.protocol)

        weakset, depickled1, depickled2, depickled3 = result

        self.assertEqual(depickled1.x, 1)
        self.assertEqual(depickled2.x, 2)
        self.assertEqual(depickled3.x, 3)
        self.assertEqual(len(weakset), 2)

        self.assertEqual(set(weakset), {depickled1, depickled2})

    def test_function_module_name(self):
        func = lambda x: x
        cloned = pickle_depickle(func, protocol=self.protocol)
        self.assertEqual(cloned.__module__, func.__module__)

    def test_function_qualname(self):
        def func(x):
            return x
        # Default __qualname__ attribute (Python 3 only)
        if hasattr(func, '__qualname__'):
            cloned = pickle_depickle(func, protocol=self.protocol)
            self.assertEqual(cloned.__qualname__, func.__qualname__)

        # Mutated __qualname__ attribute
        func.__qualname__ = '<modifiedlambda>'
        cloned = pickle_depickle(func, protocol=self.protocol)
        self.assertEqual(cloned.__qualname__, func.__qualname__)

    def test_builtin_type__new__(self):
        # Functions occasionally take the __new__ of these types as default
        # parameters for factories.  For example, on Python 3.3,
        # `tuple.__new__` is a default value for some methods of namedtuple.
        for t in list, tuple, set, frozenset, dict, object:
            cloned = pickle_depickle(t.__new__, protocol=self.protocol)
            self.assertTrue(cloned is t.__new__)

    def test_interactively_defined_function(self):
        # Check that callables defined in the __main__ module of a Python
        # script (or jupyter kernel) can be pickled / unpickled / executed.
        code = """\
        from testutils import subprocess_pickle_echo

        CONSTANT = 42

        class Foo(object):

            def method(self, x):
                return x

        foo = Foo()

        def f0(x):
            return x ** 2

        def f1():
            return Foo

        def f2(x):
            return Foo().method(x)

        def f3():
            return Foo().method(CONSTANT)

        def f4(x):
            return foo.method(x)

        cloned = subprocess_pickle_echo(lambda x: x**2, protocol={protocol})
        assert cloned(3) == 9

        cloned = subprocess_pickle_echo(f0, protocol={protocol})
        assert cloned(3) == 9

        cloned = subprocess_pickle_echo(Foo, protocol={protocol})
        assert cloned().method(2) == Foo().method(2)

        cloned = subprocess_pickle_echo(Foo(), protocol={protocol})
        assert cloned.method(2) == Foo().method(2)

        cloned = subprocess_pickle_echo(f1, protocol={protocol})
        assert cloned()().method('a') == f1()().method('a')

        cloned = subprocess_pickle_echo(f2, protocol={protocol})
        assert cloned(2) == f2(2)

        cloned = subprocess_pickle_echo(f3, protocol={protocol})
        assert cloned() == f3()

        cloned = subprocess_pickle_echo(f4, protocol={protocol})
        assert cloned(2) == f4(2)
        """.format(protocol=self.protocol)
        assert_run_python_script(textwrap.dedent(code))

    def test_interactively_defined_global_variable(self):
        # Check that callables defined in the __main__ module of a Python
        # script (or jupyter kernel) correctly retrieve global variables.
        code_template = """\
        from testutils import subprocess_pickle_echo
        from cloudpickle import dumps, loads

        def local_clone(obj, protocol=None):
            return loads(dumps(obj, protocol=protocol))

        VARIABLE = "default_value"

        def f0():
            global VARIABLE
            VARIABLE = "changed_by_f0"

        def f1():
            return VARIABLE

        cloned_f0 = {clone_func}(f0, protocol={protocol})
        cloned_f1 = {clone_func}(f1, protocol={protocol})
        pickled_f1 = dumps(f1, protocol={protocol})

        # Change the value of the global variable
        cloned_f0()

        # Ensure that the global variable is the same for another function
        result_f1 = cloned_f1()
        assert result_f1 == "changed_by_f0", result_f1

        # Ensure that unpickling the global variable does not change its value
        result_pickled_f1 = loads(pickled_f1)()
        assert result_pickled_f1 == "changed_by_f0", result_pickled_f1
        """
        for clone_func in ['local_clone', 'subprocess_pickle_echo']:
            code = code_template.format(protocol=self.protocol,
                                        clone_func=clone_func)
            assert_run_python_script(textwrap.dedent(code))

    def test_closure_interacting_with_a_global_variable(self):
        global _TEST_GLOBAL_VARIABLE
        assert _TEST_GLOBAL_VARIABLE == "default_value"
        orig_value = _TEST_GLOBAL_VARIABLE
        try:
            def f0():
                global _TEST_GLOBAL_VARIABLE
                _TEST_GLOBAL_VARIABLE = "changed_by_f0"

            def f1():
                return _TEST_GLOBAL_VARIABLE

            cloned_f0 = cloudpickle.loads(cloudpickle.dumps(
                f0, protocol=self.protocol))
            cloned_f1 = cloudpickle.loads(cloudpickle.dumps(
                f1, protocol=self.protocol))
            pickled_f1 = cloudpickle.dumps(f1, protocol=self.protocol)

            # Change the value of the global variable
            cloned_f0()
            assert _TEST_GLOBAL_VARIABLE == "changed_by_f0"

            # Ensure that the global variable is the same for another function
            result_cloned_f1 = cloned_f1()
            assert result_cloned_f1 == "changed_by_f0", result_cloned_f1
            assert f1() == result_cloned_f1

            # Ensure that unpickling the global variable does not change its
            # value
            result_pickled_f1 = cloudpickle.loads(pickled_f1)()
            assert result_pickled_f1 == "changed_by_f0", result_pickled_f1
        finally:
            _TEST_GLOBAL_VARIABLE = orig_value

    def test_pickle_reraise(self):
        for exc_type in [Exception, ValueError, TypeError, RuntimeError]:
            obj = RaiserOnPickle(exc_type("foo"))
            with pytest.raises((exc_type, pickle.PicklingError)):
                cloudpickle.dumps(obj)

    def test_unhashable_function(self):
        d = {'a': 1}
        depickled_method = pickle_depickle(d.get)
        self.assertEqual(depickled_method('a'), 1)
        self.assertEqual(depickled_method('b'), None)

    def test_itertools_count(self):
        counter = itertools.count(1, step=2)

        # advance the counter a bit
        next(counter)
        next(counter)

        new_counter = pickle_depickle(counter, protocol=self.protocol)

        self.assertTrue(counter is not new_counter)

        for _ in range(10):
            self.assertEqual(next(counter), next(new_counter))

    def test_wraps_preserves_function_name(self):
        script = '''
        from tests.cloudpickle_test import pickle_depickle
        from functools import wraps


        def f():
            pass

        @wraps(f)
        def g():
            f()

        f2 = pickle_depickle(g)

        assert f2.__name__ == f.__name__
        '''
        assert_run_python_script(textwrap.dedent(script))

    def test_wraps_preserves_function_doc(self):
        script = '''
        from tests.cloudpickle_test import pickle_depickle
        from functools import wraps

        def f():
            """42"""
            pass

        @wraps(f)
        def g():
            f()

        f2 = pickle_depickle(g)

        assert f2.__doc__ == f.__doc__
        '''
        assert_run_python_script(textwrap.dedent(script))

    @unittest.skipIf(sys.version_info < (3, 7),
                     """This syntax won't work on py2 and pickling annotations
                     isn't supported for py36 and below.""")
    def test_wraps_preserves_function_annotations(self):
        script = '''
        from tests.cloudpickle_test import pickle_depickle
        from functools import wraps

        def f(x):
            pass

        f.__annotations__ = {'x': 1, 'return': float}

        @wraps(f)
        def g(x):
            f(x)

        f2 = pickle_depickle(g)

        assert f2.__annotations__ == f.__annotations__
        '''
        assert_run_python_script(textwrap.dedent(script))


class Protocol2CloudPickleTest(CloudPickleTest):

    protocol = 2


if __name__ == '__main__':
    unittest.main()
