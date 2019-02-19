"""
New, fast version of the Cloudpickler.

This new Cloudpickler class can now extend the fast C Pickler instead of the
previous pythonic Pickler. Because this functionality is only available for
python versions 3.8+, a lot of backward-compatibilty code is also removed.
"""
import abc
import dis
import io
import opcode
import pickle
import sys
import types

from _pickle import Pickler

# XXX: Uncovered code in cloudpickle is currently removed, as they lack a
# specific use case justifying their presence. Functions/Methods removed:
# - _restore_attr
# - _get_module_builtins
# - print_exec
# - _modules_to_main
# - _gen_ellipsis
# - everyting after (if obj.__dict__) in save_global

# cloudpickle is meant for inter process communication: we expect all
# communicating processes to run the same Python version hence we favor
# communication speed over compatibility:
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL


# relevant opcodes, used to detect global variables manipulation
# XXX: I think STORE_GLOBAL can actually be removed.
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)


# map a type to its name in the types module.
_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k


# Shorthands similar to pickle.dump/pickle.dumps

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
    file = io.BytesIO()
    try:
        cp = CloudPickler(file, protocol=protocol)
        cp.dump(obj)
        return file.getvalue()
    finally:
        file.close()


# Utility functions introspecting objects to extract useful properties about
# them.

def islambda(func):
    return getattr(func, '__name__') == '<lambda>'


def _is_dynamic(module):
    """ Check if the module is importable by name

    Notable exceptions include modules created dynamically using
    types.ModuleType
    """
    # Quick check: module that have __file__ attribute are not dynamic modules.
    if hasattr(module, '__file__'):
        return False

    # XXX: there used to be backwad compat code for python 2 here.
    if hasattr(module, '__spec__'):
        return module.__spec__ is None


def extract_code_globals(code, globals):
    """
    Find all globals names read or written to by codeblock co
    """
    # XXX: there used to be a cache lookup based on the code object to get its
    # corresponding global variable names. I removed it for the first version,
    # I don't know if it is worth keeping it.
    code_globals = {}

    # PyPy "builtin-code" do not have this structure
    if hasattr(code, 'co_names'):
        instructions = dis.get_instructions(code)
        for ins in instructions:
            varname = ins.argval
            if ins.opcode in GLOBAL_OPS and varname in globals:
                code_globals.add(varname)

        # co.co_consts refers to any constant variable used by co.
        # lines such as print("foo") or a = 1 will result in a new addition to
        # the co_consts tuple ("foo" or 1).
        # However, name resolution is done at run-time, so assignment of the
        # form a = b will not yield a new item in co_consts (as the compiler
        # has no idea what b is at declaration time).

        # Declaring a function inside another one using the "def ..." syntax
        # generates a constant code object corresonding to the one of the
        # nested function's. This code object is added into the co_consts
        # attribute of the enclosing's function code. As the nested function
        # may itself need global variables, we need to introspect its code,
        # extract its globals, (look for code object in it's co_consts
        # attribute..) and add the result to the global variables lists
        if code.co_consts:
            for c in code.co_consts:
                if isinstance(c, types.CodeType):
                    code_globals.update(extract_code_globals(c, globals))

    return code_globals


# COLLECTION OF OBJECTS __getnewargs__-like methods
# -------------------------------------------------

def function_getnewargs(func, globals_ref):
    code = func.__code__

    # base_globals represents the future global namespace of func at
    # unpickling time. Looking it up and storing it in globals_ref allow
    # functions sharing the same globals at pickling time to also
    # share them once unpickled, at one condition: since globals_ref is
    # an attribute of a Cloudpickler instance, and that a new CloudPickler is
    # created each time pickle.dump or pickle.dumps is called, functions
    # also need to be saved within the same invokation of
    # cloudpickle.dump/cloudpickle.dumps
    # (for example: cloudpickle.dumps([f1, f2])). There
    # is no such limitation when using Cloudpickler.dump, as long as the
    # multiple invokations are bound to the same Cloudpickler.
    base_globals = globals_ref.setdefault(id(func.__globals__), {})

    # Do not bind the free variables before the function is created to avoid
    # infinite recursion.
    closure = tuple(types.CellType() for _ in range(len(code.co_freevars)))

    return code, base_globals, None, None, closure


# COLLECTION OF OBJECTS RECONSTRUCTORS
# ------------------------------------

# Builtin types are types defined in the python language source code, that are
# not defined in an importable python module (Lib/* for pure python module,
# Modules/* for C-implemented modules). The most wildely used ones (such as
# tuple, dict, list) are made accessible in any interpreter session by exposing
# them in the builtin namespace at startup time.

# By construction, builtin types do not have a module. Trying to access their
# __module__ attribute will default to 'builtins', that only contains builtin
# types accessible at interpreter startup. Therefore, trying to pickle the
# other ones using classic module attribute lookup instructions will fail.

# Fortunately, many non-accessible builtin-types are mirrored in the types
# module. For those types, we pickle the function builtin_type_reconstructor
# instead, that contains instruction to look them up via the types module.
def builtin_type_reconstructor(name):
    """Return a builtin-type using attribute lookup from the types module"""
    return getattr(types, name)


# XXX: what does "not working as desired" means?
# hack for __import__ not working as desired
def module_reconstructor(name):
    __import__(name)
    return sys.modules[name]


def dynamic_module_reconstructor(name, vars):
    mod = types.ModuleType(name)
    mod.__dict__.update(vars)
    return mod


def dynamic_class_reconstructor(
        tp, name, bases, tp_kwargs, registered_subclasses):
    cls = tp(name, bases, tp_kwargs)

    # Programatically register back the subclasses after creation
    for subcls in registered_subclasses:
        cls.register(subcls)


# COLLECTION OF OBJECTS STATE GETTERS
# -----------------------------------
def function_getstate(func):
    # * Put func's dynamic attributes (stored in func.__dict__) in state. These
    #   attributes will be restored at unpickling time using
    #   f.__dict__.update(state)
    # * Put func's members into slotstate. Such attributes will be restored at
    #   unpickling time by iterating over slotstate and calling setattr(func,
    #   slotname, slotvalue)
    slotstate = {
        '__name__': func.__name__,
        '__qualname__': func.__qualname__,
        '__annotations__': func.__annotations__,
        '__kwdefaults__': func.__kwdefaults__,
        '__defaults__': func.__defaults__,
        '__module__': func.__module__,
        '__doc__': func.__doc__,
        '__closure__': func.__closure__,
        '__globals__': extract_code_globals(func.__code__, func.__globals__)
    }
    state = func.__dict__
    return state, slotstate


# COLLECTIONS OF OBJECTS REDUCERS
# -------------------------------
# A reducer is a function taking a single argument (obj), and that returns a
# tuple with all the necessary data to re-construct obj. Apart from a few
# exceptions (list, dicts, bytes, ints, etc.), a reducer is necessary to
# correclty pickle an object.
# While many built-in objects (Exceptions objects, instances of the "object"
# class, etc), are shipped with their own built-in reducer (invoked using
# obj.__reduce__), some do not. The following methods were created to "fill
# these holes".
def cell_reduce(obj):
    """Cell (containing values of a function's free variables) reducer"""
    try:
        obj.cell_contents
    except ValueError:  # cell is empty
        return types.CellType, ()
    else:
        return types.CellType, (obj.cell_contents, )


def module_reduce(obj):
    """Module reducer"""
    if _is_dynamic(obj):
        return dynamic_module_reconstructor, (obj.__name__, vars(obj))
    else:
        return module_reconstructor, obj.__name__,


def dynamic_function_reduce(func, globals_ref):
    """Reduce a function that is not pickleable via attribute loookup.
    """
    # XXX: should globals_ref be a global variable instead? The reason is
    # purely cosmetic. There is no risk of references leaking, we would have to
    # limit the growing of globals_ref, by making it a lru cache for example.
    newargs = function_getnewargs(func, globals_ref)
    state = function_getstate(func)
    return types.FunctionType, newargs, state


def dynamic_class_reduce(self, obj):
    """
    Save a class that can't be stored as module global.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from global modules.
    """
    # XXX: This code is nearly untouch with regards to the legacy cloudpickle.
    # It is pretty and hard to understand. Maybe refactor it by dumping
    # potential python2 specific code and making a trading off optimizations in
    # favor of readbility.
    clsdict = dict(obj.__dict__)  # copy dict proxy to a dict
    clsdict.pop('__weakref__', None)

    # XXX: I am trying to add the abc-registered subclasses into the class
    # reconstructor, because using save_reduce semantics prevents us to perform
    # any other operation than state updating after obj is created.

    # I may encounter reference cycles, although there seems to be checks
    # preventing this to happen.
    if "_abc_impl" in clsdict:
        (registry, _, _, _) = abc._get_dump(obj)
        subclasses = [subclass_weakref() for subclass_weakref in registry]
        clsdict.pop("_abc_impl")

    # On PyPy, __doc__ is a readonly attribute, so we need to include it in
    # the initial skeleton class.  This is safe because we know that the
    # doc can't participate in a cycle with the original class.
    type_kwargs = {'__doc__': clsdict.pop('__doc__', None)}

    if hasattr(obj, "__slots__"):
        type_kwargs['__slots__'] = obj.__slots__
        # Pickle string length optimization: member descriptors of obj are
        # created automatically from obj's __slots__ attribute, no need to
        # save them in obj's state
        if isinstance(obj.__slots__, str):
            clsdict.pop(obj.__slots__)
        else:
            for k in obj.__slots__:
                clsdict.pop(k, None)

    # If type overrides __dict__ as a property, include it in the type kwargs.
    # In Python 2, we can't set this attribute after construction.
    # XXX: when does this happen? Is this python-2 specific?
    __dict__ = clsdict.pop('__dict__', None)
    if isinstance(__dict__, property):
        type_kwargs['__dict__'] = __dict__
        __dict__ = None

    return (
        dynamic_module_reconstructor,
        (type(obj), obj.__name__, obj.__bases__, type_kwargs, subclasses),
        (__dict__, clsdict)  # state, slotstate structure.
    )


# Arbitration between builtin-save method and user-defined callbacks
# ------------------------------------------------------------------
# This set of functions aim at deciding whether an object can be properly
# pickler by the c Pickler, or if it needs to be serialized using cloudpickle's
# reducers.
def save_function(self, obj, name=None):
    """ Registered with the dispatch to handle all function types.

    Determines what kind of function obj is (e.g. lambda, defined at
    interactive prompt, etc) and handles the pickling appropriately.
    """
    if obj in _BUILTIN_TYPE_CONSTRUCTORS:
        # We keep a special-cased cache of built-in type constructors at
        # global scope, because these functions are structured very
        # differently in different python versions and implementations (for
        # example, they're instances of types.BuiltinFunctionType in
        # CPython, but they're ordinary types.FunctionType instances in
        # PyPy).
        #
        # If the function we've received is in that cache, we just
        # serialize it as a lookup into the cache.
        return _BUILTIN_TYPE_CONSTRUCTORS[obj], ()

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
        if lookedup_by_name is obj:
            # default to save_global
            return NotImplementedError

    # a builtin_function_or_method which comes in as an attribute of some
    # object (e.g., itertools.chain.from_iterable) will end
    # up with modname "__main__" and so end up here. But these functions
    # have no __code__ attribute in CPython, so the handling for
    # user-defined functions below will fail.
    # So we pickle them here using save_reduce; have to do it differently
    # for different python versions.
    if not hasattr(obj, '__code__'):
        if PY3:  # pragma: no branch
            rv = obj.__reduce_ex__(self.proto)
        else:
            if hasattr(obj, '__self__'):
                rv = (getattr, (obj.__self__, name))
            else:
                raise pickle.PicklingError("Can't pickle %r" % obj)
        return rv

    # if func is lambda, def'ed at prompt, is in main, or is nested, then
    # we'll pickle the actual function object rather than simply saving a
    # reference (as is done in default pickler), via save_function_tuple.
    if (islambda(obj)
            or getattr(obj.__code__, 'co_filename', None) == '<stdin>'
            or themodule is None):
        return self.save_function_tuple(obj)
    else:
        # func is nested
        if lookedup_by_name is None or lookedup_by_name is not obj:
            return self.save_function_tuple(obj)


def hook(pickler, obj):
    """Custom reducing instructions for un-picklable functions and classes
    """
    # Classes deriving from custom, dynamic metaclasses won't get caught inside
    # the hook_dispatch dict. In the legacy cloudpickle, this was not really a
    # problem because not being present in the dispatch table meant falling
    # back to save_global, which was already overriden by cloudpickle. Using
    # the c pickler, save_global cannot be overriden, so we have manually check
    # is obj's comes from a custom metaclass, and in this case, direct the
    # object to save_global.
    t = type(obj)

    try:
        has_custom_metaclass = issubclass(t, type)
    except TypeError:  # t is not a class (old Boost; see SF #502085)
        has_custom_metaclass = False
    if has_custom_metaclass:
        return pickler.save_global(obj)

    # Else, do a classic lookup on the hook dispatch
    reducer = CloudPickler.callback_dispatch.get(t)
    if reducer is None:
        return NotImplementedError
    else:
        return reducer(pickler, obj)


class CloudPickler(Pickler):
    """Fast C Pickler extension with additional reducing routines


   Cloudpickler's extensions exist into into:

   * it's dispatch_table containing methods that are called only if ALL
     built-in saving functions were previously discarded.
   * it's callback_dispatch, containing methods that are called only if ALL
     built-in saving functions except save_global were previously discarded.

    Both tables contains reducers, that take a single argument (obj), and
    preturn a tuple with all the necessary data to re-construct obj.

    """
    dispatch = {}
    dispatch[types.CellType] = cell_reduce
    dispatch[types.MethodType] = module_reduce
    callback_dispatch = {}

    def __init__(self, file, protocol=None):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        Pickler.__init__(self, file, protocol=protocol)
        # map functions __globals__ attribute ids, to ensure that functions
        # sharing the same global namespace at pickling time also share their
        # global namespace at unpickling time.
        self.globals_ref = {}
        self.dispatch_table = self.dispatch
        self.global_hook = hook
        self.proto = int(protocol)

    def dump(self, obj):
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = ("Could not pickle object as excessively deep recursion "
                       "required.")
                raise pickle.PicklingError(msg)
            else:
                raise
