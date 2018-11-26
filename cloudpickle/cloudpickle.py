"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function

import dis
import sys
import types
import opcode
import pickle
import struct
import weakref
import importlib
import itertools
import traceback

from pickle import _Pickler as Pickler
from io import BytesIO as StringIO

# cloudpickle is meant for inter process communication: we expect all
# communicating processes to run the same Python version hence we favor
# communication speed over compatibility:
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL


types.ClassType = type



# relevant opcodes
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG


def islambda(func):
    return getattr(func, '__name__') == '<lambda>'


_BUILTIN_TYPE_NAMES = {
        types.CodeType: 'CodeType',
        types.MethodType: 'MethodType'
        }


def _builtin_type(name):
    return getattr(types, name)


def _walk_global_ops(code):
    """
    Yield (opcode, argument number) tuples for all
    global-referencing instructions in *code*.
    """
    for instr in dis.get_instructions(code):
        op = instr.opcode
        if op in GLOBAL_OPS:
            yield op, instr.arg


class CloudPickler(Pickler):

    dispatch = Pickler.dispatch.copy()

    def __init__(self, file, protocol=None):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        Pickler.__init__(self, file, protocol=protocol)
        # set of modules to unpickle
        self.modules = set()
        # map ids to dictionary. used to ensure that functions can share global env
        self.globals_ref = {}

    def dump(self, obj):
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = """Could not pickle object as excessively deep recursion required."""
                raise pickle.PicklingError(msg)
            else:
                raise

    def save_module(self, obj):
        """
        Save a module as an import
        """
        self.modules.add(obj)
        self.save_reduce(subimport, (obj.__name__,), obj=obj)

    dispatch[types.ModuleType] = save_module

    def save_codeobject(self, obj):
        """
        Save a code object
        """
        args = (
            obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames,
            obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
            obj.co_cellvars
        )
        self.save_reduce(types.CodeType, args, obj=obj)

    dispatch[types.CodeType] = save_codeobject

    def save_function(self, obj, name=None):
        """ Registered with the dispatch to handle all function types.

        Determines what kind of function obj is (e.g. lambda, defined at
        interactive prompt, etc) and handles the pickling appropriately.
        """
        write = self.write

        if name is None:
            name = obj.__name__
        try:
            # whichmodule() could fail, see
            # https://bitbucket.org/gutworth/six/issues/63/importing-six-breaks-pickling
            modname = pickle.whichmodule(obj, name)
        except Exception:
            modname = None
        # print('which gives %s %s %s' % (modname, obj, name))
        try:
            themodule = sys.modules[modname]
        except KeyError:
            # eval'd items such as namedtuple give invalid items for their function __module__
            modname = '__main__'

        if modname == '__main__':
            themodule = None

        try:
            lookedup_by_name = getattr(themodule, name, None)
        except Exception:
            lookedup_by_name = None

        if themodule:
            self.modules.add(themodule)
            if lookedup_by_name is obj:
                return self.save_global(obj, name)

        # a builtin_function_or_method which comes in as an attribute of some
        # object (e.g., itertools.chain.from_iterable) will end
        # up with modname "__main__" and so end up here. But these functions
        # have no __code__ attribute in CPython, so the handling for
        # user-defined functions below will fail.
        # So we pickle them here using save_reduce; have to do it differently
        # for different python versions.
        if not hasattr(obj, '__code__'):
            rv = obj.__reduce_ex__(self.proto)
            return self.save_reduce(obj=obj, *rv)

        # if func is lambda, def'ed at prompt, is in main, or is nested, then
        # we'll pickle the actual function object rather than simply saving a
        # reference (as is done in default pickler), via save_function_tuple.
        if (islambda(obj)
                or getattr(obj.__code__, 'co_filename', None) == '<stdin>'
                or themodule is None):
            self.save_function_tuple(obj)
            return
        else:
            # func is nested
            if lookedup_by_name is None or lookedup_by_name is not obj:
                self.save_function_tuple(obj)
                return

        if obj.__dict__:
            # essentially save_reduce, but workaround needed to avoid recursion
            self.save(_restore_attr)
            write(pickle.MARK + pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
            self.save(obj.__dict__)
            write(pickle.TUPLE + pickle.REDUCE)
        else:
            write(pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)

    dispatch[types.FunctionType] = save_function

    def _save_subimports(self, code, top_level_dependencies):
        """
        Ensure de-pickler imports any package child-modules that
        are needed by the function
        """

        # check if any known dependency is an imported package
        for x in top_level_dependencies:
            if isinstance(x, types.ModuleType) and hasattr(x, '__package__') and x.__package__:
                # check if the package has any currently loaded sub-imports
                prefix = x.__name__ + '.'
                # A concurrent thread could mutate sys.modules,
                # make sure we iterate over a copy to avoid exceptions
                for name in list(sys.modules):
                    # Older versions of pytest will add a "None" module to sys.modules.
                    if name is not None and name.startswith(prefix):
                        # check whether the function can address the sub-module
                        tokens = set(name[len(prefix):].split('.'))
                        if not tokens - set(code.co_names):
                            # ensure unpickler executes this import
                            self.save(sys.modules[name])
                            # then discards the reference to it
                            self.write(pickle.POP)

    def save_function_tuple(self, func):
        """  Pickles an actual func object.

        A func comprises: code, globals, defaults, closure, and dict.  We
        extract and save these, injecting reducing functions at certain points
        to recreate the func object.  Keep in mind that some of these pieces
        can contain a ref to the func itself.  Thus, a naive save on these
        pieces could trigger an infinite loop of save's.  To get around that,
        we first create a skeleton func object using just the code (this is
        safe, since this won't contain a ref to the func), and memoize it as
        soon as it's created.  The other stuff can then be filled in later.
        """
        save = self.save
        write = self.write

        code, f_globals, defaults, closure_values, dct, base_globals = self.extract_func_data(func)  # noqa
        if closure_values is not None:
            raise pickle.PicklingError('cannot pickle a function with a '
                                       'non-empty closure')

        save(_fill_function)  # skeleton function updater
        write(pickle.MARK)    # beginning of tuple that _fill_function expects

        self._save_subimports(code, f_globals.values())

        # create a skeleton function object and memoize it
        save(_make_skel_func)
        save((
            code,
            base_globals,
        ))
        write(pickle.REDUCE)
        self.memoize(func)

        # save the rest of the func data needed by _fill_function
        state = {
            'globals': f_globals,
            'defaults': defaults,
            'dict': dct,
            'closure_values': closure_values,
            'module': func.__module__,
            'name': func.__name__,
            'doc': func.__doc__,
            'annotations': func.__annotations__,
            'qualname': func.__qualname__
        }
        save(state)
        write(pickle.TUPLE)
        write(pickle.REDUCE)  # applies _fill_function on the tuple

    _extract_code_globals_cache = weakref.WeakKeyDictionary()

    @classmethod
    def extract_code_globals(cls, co):
        """
        Find all globals names read or written to by codeblock co
        """
        out_names = cls._extract_code_globals_cache.get(co)
        if out_names is None:
            try:
                names = co.co_names
            except AttributeError:
                # PyPy "builtin-code" object
                out_names = set()
            else:
                out_names = {names[oparg] for _, oparg in _walk_global_ops(co)}

                # see if nested function have any global refs
                if co.co_consts:
                    for const in co.co_consts:
                        if type(const) is types.CodeType:
                            out_names |= cls.extract_code_globals(const)

            cls._extract_code_globals_cache[co] = out_names

        return out_names

    def extract_func_data(self, func):
        """
        Turn the function into a tuple of data necessary to recreate it:
            code, globals, defaults, closure_values, dict
        """
        code = func.__code__

        # extract all global ref's
        func_global_refs = self.extract_code_globals(code)

        # process all variables referenced by global environment
        f_globals = {}
        for var in func_global_refs:
            if var in func.__globals__:
                f_globals[var] = func.__globals__[var]

        # defaults requires no processing
        defaults = func.__defaults__

        # process closure
        closure = func.__closure__

        # save the dict
        dct = func.__dict__

        base_globals = self.globals_ref.get(id(func.__globals__), None)
        if base_globals is None:
            # For functions defined in a well behaved module use
            # vars(func.__module__) for base_globals. This is necessary to
            # share the global variables across multiple pickled functions from
            # this module.
            if func.__module__ is not None:
                base_globals = func.__module__
            else:
                base_globals = {}
        self.globals_ref[id(func.__globals__)] = base_globals

        return (code, f_globals, defaults, closure, dct, base_globals)

    def save_global(self, obj, name=None, pack=struct.pack):
        """
        Save a "global".

        The name of this method is somewhat misleading: all types get
        dispatched here.
        """
        if obj is type(None):
            return self.save_reduce(type, (None,), obj=obj)
        elif obj is type(NotImplemented):
            return self.save_reduce(type, (NotImplemented,), obj=obj)

        try:
            return Pickler.save_global(self, obj, name=name)
        except Exception:
            if obj.__module__ == "__builtin__" or obj.__module__ == "builtins":
                if obj in _BUILTIN_TYPE_NAMES:
                    return self.save_reduce(
                        _builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)

            raise

    dispatch[type] = save_global


# Shorthands for legacy support

def dump(obj, file, protocol=None):
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    CloudPickler(file, protocol=protocol).dump(obj)


def dumps(obj, protocol=None):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    file = StringIO()
    try:
        cp = CloudPickler(file, protocol=protocol)
        cp.dump(obj)
        return file.getvalue()
    finally:
        file.close()


# including pickles unloading functions in this namespace
load = pickle.load
loads = pickle.loads


# hack for __import__ not working as desired
def subimport(name):
    __import__(name)
    return sys.modules[name]


# restores function attributes
def _restore_attr(obj, attr):
    for key, val in attr.items():
        setattr(obj, key, val)
    return obj


def print_exec(stream):
    ei = sys.exc_info()
    traceback.print_exception(ei[0], ei[1], ei[2], None, stream)


def _fill_function(*args):
    """Fills in the rest of function data into the skeleton function object

    The skeleton itself is create by _make_skel_func().
    """
    func, state = args

    # Only set global variables that do not exist.
    for k, v in state['globals'].items():
        if k not in func.__globals__:
            func.__globals__[k] = v

    func.__defaults__ = state['defaults']
    func.__dict__ = state['dict']
    func.__annotations__ = state['annotations']
    func.__doc__ = state['doc']
    func.__name__ = state['name']
    func.__module__ = state['module']
    func.__qualname__ = state['qualname']

    return func


def _make_skel_func(code, base_globals=None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    if base_globals is None:
        base_globals = {}
    elif isinstance(base_globals, str):
        try:
            # First try to reuse the globals from the module containing the
            # function. If it is not possible to retrieve it, fallback to an
            # empty dictionary.
            base_globals = vars(importlib.import_module(base_globals))
        except ImportError:
            base_globals = {}

    base_globals['__builtins__'] = __builtins__

    return types.FunctionType(code, base_globals, None, None, None)
