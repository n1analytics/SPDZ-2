"""
This module defines all types availabe in high-level programs.  These
include basic types such as secret integers or floating-point numbers
and container types. A single instance of the former uses one or more
so-called registers in the virtual machine while the latter use the
so-called memory.  For every register type, there is a corresponding
dedicated memory.

Registers are used for computation, allocated on an ongoing basis,
and thread-specific. The memory is allocated statically and shared
between threads. This means that memory-based types such as
:py:class:`Array` can be used to transfer information between threads.
Note that creating memory-based types outside the main thread is not
supported.

If viewing this documentation in processed form, many function signatures
appear generic because of the use of decorators. See the source code for the
correct signature.

Basic types
-----------

Basic types contain many special methods such as :py:func:`__add__`. This is
used for operator overloading in Python. It is not recommend to use
them, use the plain operators instead, such as ``+`` instead of
:py:func:`__add__`. See
https://docs.python.org/3/reference/datamodel.html#special-method-names
for a translation to operators.

In some operations such as
secure comparison, the secure computation protocols allows for more
parameters than just the operands which influence the performance. In
this case, we provide an alias for better code readability. For
example, :meth:`sint.greater_than` is an alias of
:py:meth:`sint.__gt__`. When using operator overloading, the
parameters default to the globally defined ones.

Methods of basic types generally return instances of the respective type.

Note that the data model of Python operates with reverse operators
such as :py:func:`__radd__`. This means that if for the usual operator of the
first operand does not support the second operand, the reverse
operator of the second operand is used. For example,
:py:meth:`_clear.__sub__` does not support secret values as second
operand but :py:meth:`_secret.__rsub__` does support clear values, so
``cint(3) - sint(2)`` will result in a secret integer of value 1.

.. autosummary::
   :nosignatures:

   sint
   cint
   regint
   sfix
   cfix
   sfloat
   sgf2n
   cgf2n

Container types
---------------

.. autosummary::
   :nosignatures:

   MemValue
   Array
   Matrix
   MultiArray

"""

from Compiler.program import Tape
from Compiler.exceptions import *
from Compiler.instructions import *
from Compiler.instructions_base import *
from .floatingpoint import two_power
from . import comparison, floatingpoint
import math
from . import util
from . import instructions
from .util import is_zero, is_one
import operator
from functools import reduce
import re


class ClientMessageType:
    """ Enum to define type of message sent to external client. Each may be array of length n."""
    # No client message type to be sent, for backwards compatibility - virtual machine relies on this value
    NoType = 0
    # 3 x sint x n
    TripleShares = 1
    # 1 x cint x n
    ClearModpInt = 2
    # 1 x regint x n
    Int32 = 3
    # 1 x cint (fixed point left shifted by precision) x n
    ClearModpFix = 4


class MPCThread(object):
    def __init__(self, target, name, args = [], runtime_arg = 0,
                 single_thread = False):
        """ Create a thread from a callable object. """
        if not callable(target):
            raise CompilerError('Target %s for thread %s is not callable' % (target,name))
        self.name = name
        self.tape = Tape(program.name + '-' + name, program)
        self.target = target
        self.args = args
        self.runtime_arg = runtime_arg
        self.running = 0
        self.tape_handle = program.new_tape(target, args, name,
                                            single_thread=single_thread)
        self.run_handles = []
    
    def start(self, runtime_arg = None):
        self.running += 1
        self.run_handles.append(program.run_tape(self.tape_handle, \
                                           runtime_arg or self.runtime_arg))
    
    def join(self):
        if not self.running:
            raise CompilerError('Thread %s is not running' % self.name)
        self.running -= 1
        program.join_tape(self.run_handles.pop(0))


def copy_doc(a, b):
    try:
        a.__doc__ = b.__doc__
    except:
        pass

def no_doc(operation):
    def wrapper(*args, **kwargs):
        return operation(*args, **kwargs)
    return wrapper

def vectorize(operation):
    def vectorized_operation(self, *args, **kwargs):
        if len(args):
            from .GC.types import bits
            if (isinstance(args[0], Tape.Register) or isinstance(args[0], sfloat)) \
                    and not isinstance(args[0], bits) \
                    and args[0].size != self.size:
                raise CompilerError('Different vector sizes of operands: %d/%d'
                                    % (self.size, args[0].size))
        set_global_vector_size(self.size)
        try:
            res = operation(self, *args, **kwargs)
        finally:
            reset_global_vector_size()
        return res
    copy_doc(vectorized_operation, operation)
    return vectorized_operation

def vectorize_max(operation):
    def vectorized_operation(self, *args, **kwargs):
        size = self.size
        for arg in args:
            try:
                size = max(size, arg.size)
            except AttributeError:
                pass
        set_global_vector_size(size)
        try:
            res = operation(self, *args, **kwargs)
        finally:
            reset_global_vector_size()
        return res
    copy_doc(vectorized_operation, operation)
    return vectorized_operation

def vectorized_classmethod(function):
    def vectorized_function(cls, *args, **kwargs):
        size = None
        if 'size' in kwargs:
            size = kwargs.pop('size')
        if size:
            set_global_vector_size(size)
            try:
                res = function(cls, *args, **kwargs)
            finally:
                reset_global_vector_size()
        else:
            res = function(cls, *args, **kwargs)
        return res
    copy_doc(vectorized_function, function)
    return classmethod(vectorized_function)

def vectorize_init(function):
    def vectorized_init(*args, **kwargs):
        size = None
        if len(args) > 1 and (isinstance(args[1], Tape.Register) or \
                    isinstance(args[1], sfloat)):
            size = args[1].size
            if 'size' in kwargs and kwargs['size'] is not None \
                    and kwargs['size'] != size:
                raise CompilerError('Mismatch in vector size')
        if 'size' in kwargs and kwargs['size']:
            size = kwargs['size']
        if size is not None:
            set_global_vector_size(size)
            try:
                res = function(*args, **kwargs)
            finally:
                reset_global_vector_size()
        else:
            res = function(*args, **kwargs)
        return res
    copy_doc(vectorized_init, function)
    return vectorized_init

def set_instruction_type(operation):
    def instruction_typed_operation(self, *args, **kwargs):
        set_global_instruction_type(self.instruction_type)
        try:
            res = operation(self, *args, **kwargs)
        finally:
            reset_global_instruction_type()
        return res
    copy_doc(instruction_typed_operation, operation)
    return instruction_typed_operation

def read_mem_value(operation):
    def read_mem_operation(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], MemValue):
            args = (args[0].read(),) + args[1:]
        return operation(self, *args, **kwargs)
    copy_doc(read_mem_operation, operation)
    return read_mem_operation

def inputmixed(*args):
    # helper to cover both cases
    if isinstance(args[-1], int):
        instructions.inputmixed(*args)
    else:
        instructions.inputmixedreg(*(args[:-1] + (regint.conv(args[-1]),)))

class _number(object):
    """ Number functionality. """

    def square(self):
        """ Square. """
        return self * self

    def __add__(self, other):
        """ Optimized addition.

        :param other: any compatible type """
        if is_zero(other):
            return self
        else:
            return self.add(other)

    def __mul__(self, other):
        """ Optimized multiplication.

        :param other: any compatible type """
        if is_zero(other):
            return 0
        elif is_one(other):
            return self
        else:
            return self.mul(other)

    __radd__ = __add__
    __rmul__ = __mul__

    @vectorize
    def __pow__(self, exp):
        """ Exponentation through square-and-multiply.

        :param exp: any type allowing bit decomposition """
        if isinstance(exp, int) and exp >= 0:
            if exp == 0:
                return self.__class__(1)
            exp = bin(exp)[3:]
            res = self
            for i in exp:
                res = res.square()
                if i == '1':
                    res *= self
            return res
        else:
            bits = exp.bit_decompose()
            powers = [self]
            while len(powers) < len(bits):
                powers.append(powers[-1] ** 2)
            multiplicands = [b.if_else(p, 1) for b, p in zip(bits, powers)]
            res = util.tree_reduce(operator.mul, multiplicands)
            return res

    def mul_no_reduce(self, other, res_params=None):
        return self * other

    def reduce_after_mul(self):
        return self

    def pow2(self, bit_length=None, security=None):
        return 2**self

    def min(self, other):
        """ Minimum.

        :param other: any compatible type """
        return (self < other).if_else(self, other)

    def max(self, other):
        """ Maximum.

        :param other: any compatible type """
        return (self < other).if_else(other, self)

    @classmethod
    def dot_product(cls, a, b):
        from Compiler.library import for_range_opt_multithread
        res = MemValue(cls(0))
        l = min(len(a), len(b))
        xx = [a, b]
        for i, x in enumerate((a, b)):
            if not isinstance(x, Array):
                xx[i] = Array(l, cls)
                xx[i].assign(x)
        aa, bb = xx
        @for_range_opt_multithread(None, l)
        def _(i):
            res.iadd(res.value_type.conv(aa[i] * bb[i]))
        return res.read()

    def __abs__(self):
        """ Absolute value. """
        return (self < 0).if_else(-self, self)

    @staticmethod
    def popcnt_bits(bits):
        return sum(bits)

class _int(object):
    """ Integer functionality. """

    @staticmethod
    def bit_adder(*args, **kwargs):
        return intbitint.bit_adder(*args, **kwargs)

    @staticmethod
    def ripple_carry_adder(*args, **kwargs):
        return intbitint.ripple_carry_adder(*args, **kwargs)

    def if_else(self, a, b):
        """ MUX on bit in arithmetic circuits.

        :param a/b: any type supporting the necessary operations
        :return: a if :py:obj:`self` is 1, b if :py:obj:`self` is 0, undefined otherwise
        :rtype: depending on operands, secret if any of them is """
        if hasattr(a, 'for_mux'):
            f, a, b = a.for_mux(b)
        else:
            f = lambda x: x
        return f(self * (a - b) + b)

    def cond_swap(self, a, b):
        """ Swapping in arithmetic circuits.

        :param a/b: any type supporting the necessary operations
        :return: ``(a, b)`` if :py:obj:`self` is 0, ``(b, a)`` if :py:obj:`self` is 1, and undefined otherwise
        :rtype: depending on operands, secret if any of them is """
        prod = self * (a - b)
        return a - prod, b + prod

    def bit_xor(self, other):
        """ XOR in arithmetic circuits.

        :param self/other: 0 or 1 (any compatible type)
        :return: type depends on inputs (secret if any of them is) """
        if util.is_constant(other):
            if other:
                return 1 - self
            else:
                return self
        return self + other - 2 * self * other

    def bit_or(self, other):
        """ OR in arithmetic circuits.

        :param self/other: 0 or 1 (any compatible type)
        :return: type depends on inputs (secret if any of them is) """
        if util.is_constant(other):
            if other:
                return self
            else:
                return 0
        return self + other - self * other

    def bit_and(self, other):
        """ AND in arithmetic circuits.

        :param self/other: 0 or 1 (any compatible type)
        :rtype: depending on inputs (secret if any of them is) """
        return self * other

    def bit_not(self):
        """ NOT in arithmetic circuits. """
        return 1 - self

    def half_adder(self, other):
        """ Half adder in arithmetic circuits.

        :param self/other: 0 or 1 (any compatible type)
        :return: binary sum, carry
        :rtype: depending on inputs, secret if any is """
        carry = self * other
        return self + other - 2 * carry, carry

class _bit(object):
    """ Binary functionality. """

    def bit_xor(self, other):
        """ XOR in binary circuits.

        :param self/other: 0 or 1 (any compatible type)
        :rtype: depending on inputs (secret if any of them is) """
        return self ^ other

    def bit_and(self, other):
        """ AND in binary circuits.

        :param self/other: 0 or 1 (any compatible type)
        :rtype: depending on inputs (secret if any of them is) """
        return self & other

    def bit_or(self, other):
        """ OR in binary circuits.

        :param self/other: 0 or 1 (any compatible type)
        :return: type depends on inputs (secret if any of them is) """
        return self ^ other - self & other

    def bit_not(self):
        """ NOT in binary circuits. """
        return ~self

    def half_adder(self, other):
        """ Half adder in binary circuits.

        :param self/other: 0 or 1 (any compatible type)
        :return: binary sum, carry
        :rtype: depending on inputs (secret if any of them is) """
        return self ^ other, self & other

    def carry_out(self, a, b):
        s = a ^ b
        return a ^ (s & (self ^ a))

class _gf2n(_bit):
    """ :math:`\mathrm{GF}(2^n)` functionality. """

    def if_else(self, a, b):
        """ MUX in :math:`\mathrm{GF}(2^n)` circuits. Similar to :py:meth:`_int.if_else`. """
        return b ^ self * self.hard_conv(a ^ b)

    def cond_swap(self, a, b, t=None):
        """ Swapping in :math:`\mathrm{GF}(2^n)`. Similar to :py:meth:`_int.if_else`. """
        prod = self * self.hard_conv(a ^ b)
        res = a ^ prod, b ^ prod
        if t is None:
            return res
        else:
            return tuple(t.conv(r) for r in res)

    def bit_xor(self, other):
        """ XOR in :math:`\mathrm{GF}(2^n)` circuits.

        :param self/other: 0 or 1 (any compatible type)
        :rtype: depending on inputs (secret if any of them is) """
        return self ^ other

    def bit_not(self):
        return self ^ 1

class _structure(object):
    """ Interface for type-dependent container types. """

    MemValue = classmethod(lambda cls, value: MemValue(cls.conv(value)))
    """ Type-dependent memory value. """

    @classmethod
    def Array(cls, size, *args, **kwargs):
        """ Type-dependent array. Example:

        .. code::

            a = sint.Array(10)
        """
        return Array(size, cls, *args, **kwargs)

    @classmethod
    def Matrix(cls, rows, columns, *args, **kwargs):
        """ Type-dependent matrix. Example:

        .. code::

            a = sint.Matrix(10, 10)
        """
        return Matrix(rows, columns, cls, *args, **kwargs)

    @classmethod
    def row_matrix_mul(cls, row, matrix, res_params=None):
        return sum(row[k].mul_no_reduce(matrix[k].get_vector(),
                                        res_params) \
                   for k in range(len(row))).reduce_after_mul()

    @staticmethod
    def mem_size():
        return 1

class _vec(object):
    pass

class _register(Tape.Register, _number, _structure):
    @staticmethod
    def n_elements():
        return 1

    @vectorized_classmethod
    def conv(cls, val):
        if isinstance(val, MemValue):
            val = val.read()
        if isinstance(val, cls):
            return val
        elif not isinstance(val, (_register, _vec)):
            try:
                return type(val)(cls.conv(v) for v in val)
            except TypeError:
                pass
            except CompilerError:
                pass
        return cls(val)

    @vectorized_classmethod
    @read_mem_value
    def hard_conv(cls, val):
        if type(val) == cls:
            return val
        elif not isinstance(val, _register):
            try:
                return val.hard_conv_me(cls)
            except AttributeError:
                try:
                    return type(val)(cls.hard_conv(v) for v in val)
                except TypeError:
                    pass
        return cls(val)

    @vectorized_classmethod
    @set_instruction_type
    def _load_mem(cls, address, direct_inst, indirect_inst):
        if isinstance(address, _register):
            if address.size > 1:
                size = address.size
            else:
                size = get_global_vector_size()
            res = cls(size=size)
            indirect_inst(res, cls._expand_address(address,
                                                   get_global_vector_size()))
        else:
            res = cls()
            direct_inst(res, address)
        return res

    @staticmethod
    def _expand_address(address, size):
        address = regint.conv(address)
        if size > 1 and address.size == 1:
            res = regint(size=size)
            incint(res, address, 1)
            return res
        else:
            return address

    @set_instruction_type
    def _store_in_mem(self, address, direct_inst, indirect_inst):
        if isinstance(address, _register):
            indirect_inst(self, self._expand_address(address, self.size))
        else:
            direct_inst(self, address)

    @classmethod
    def prep_res(cls, other):
        return cls()

    @classmethod
    def bit_compose(cls, bits):
        """ Compose value from bits.

        :param bits: iterable of any type implementing left shift """
        return sum(cls.conv(b) << i for i,b in enumerate(bits))

    @classmethod
    def malloc(cls, size, creator_tape=None):
        """ Allocate memory (statically).

        :param size: compile-time (int) """
        return program.malloc(size, cls, creator_tape=creator_tape)

    @classmethod
    def free(cls, addr):
        program.free(addr, cls.reg_type)

    @set_instruction_type
    def __init__(self, reg_type, val, size):
        if isinstance(val, (tuple, list)):
            size = len(val)
        super(_register, self).__init__(reg_type, program.curr_tape, size=size)
        if isinstance(val, int):
            self.load_int(val)
        elif isinstance(val, (tuple, list)):
            for i, x in enumerate(val):
                if util.is_constant(x):
                    self[i].load_int(x)
                else:
                    self[i].load_other(x)
        elif val is not None:
            self.load_other(val)

    def _new_by_number(self, i, size=1):
        res = type(self)(size=size)
        res.i = i
        res.program = self.program
        return res

    def sizeof(self):
        return self.size

    def extend(self, n):
        return self

    def expand_to_vector(self, size=None):
        if size is None:
            size = get_global_vector_size()
        if self.size == size:
            return self
        assert self.size == 1
        res = type(self)(size=size)
        for i in range(size):
            movs(res[i], self)
        return res

class _clear(_register):
    """ Clear domain-dependent type. """
    __slots__ = []
    mov = staticmethod(movc)

    @set_instruction_type
    @vectorize
    def load_other(self, val):
        if isinstance(val, type(self)):
            movc(self, val)
        else:
            self.convert_from(val)

    @vectorize
    @read_mem_value
    def convert_from(self, val):
        if not isinstance(val, regint):
            val = regint(val)
        convint(self, val)

    @set_instruction_type
    @vectorize
    def print_reg(self, comment=''):
        print_reg(self, comment)

    @set_instruction_type
    @vectorize
    def print_reg_plain(self):
        """ Output. """
        print_reg_plain(self)

    @set_instruction_type
    @vectorize
    def raw_output(self):
        raw_output(self)

    @set_instruction_type
    @read_mem_value
    @vectorize
    def clear_op(self, other, c_inst, ci_inst, reverse=False):
        cls = self.__class__
        res = self.prep_res(other)
        if isinstance(other, cls):
            c_inst(res, self, other)
        elif isinstance(other, int):
            if self.in_immediate_range(other):
                ci_inst(res, self, other)
            else:
                if reverse:
                    c_inst(res, cls(other), self)
                else:
                    c_inst(res, self, cls(other))
        else:
            return NotImplemented
        return res

    @set_instruction_type
    @read_mem_value
    @vectorize
    def coerce_op(self, other, inst, reverse=False):
        cls = self.__class__
        res = cls()
        if isinstance(other, int):
            other = cls(other)
        elif not isinstance(other, cls):
            return NotImplemented
        if reverse:
            inst(res, other, self)
        else:
            inst(res, self, other)
        return res

    def add(self, other):
        """ Addition of public values.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, addc, addci)

    def mul(self, other):
        """ Multiplication of public values.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, mulc, mulci)

    def __sub__(self, other):
        """ Subtraction of public values.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, subc, subci)

    def __rsub__(self, other):
        return self.clear_op(other, subc, subcfi, True)
    __rsub__.__doc__ = __sub__.__doc__

    def __truediv__(self, other):
        """ Field division of public values. Not available for
        computation modulo a power of two.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, divc, divci)

    def __rtruediv__(self, other):
        return self.coerce_op(other, divc, True)
    __rtruediv__.__doc__ = __truediv__.__doc__

    def __eq__(self, other):
        """ Equality check of public values.

        :param other:  convertible type (at least same as :py:obj:`self` and regint/int)
        :return: 0/1 (regint) """
        if isinstance(other, (_clear,int)):
            return regint(self) == other
        else:
            return NotImplemented

    def __ne__(self, other):
        return 1 - (self == other)
    __ne__.__doc__ = __eq__.__doc__

    def __and__(self, other):
        """ Bit-wise AND of public values.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, andc, andci)

    def __xor__(self, other):
        """ Bit-wise XOR of public values.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, xorc, xorci)

    def __or__(self, other):
        """ Bit-wise OR of public values.

        :param other: convertible type (at least same as :py:obj:`self` and regint/int) """
        return self.clear_op(other, orc, orci)

    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    def reveal(self):
        """ Identity. """
        return self


class cint(_clear, _int):
    """
    Clear integer in same domain as secure computation
    (depends on protocol).
    """
    __slots__ = []
    instruction_type = 'modp'
    reg_type = 'c'

    @vectorized_classmethod
    def read_from_socket(cls, client_id, n=1):
        """ Receive clear value(s) from client.

        :param client_id: Client id (regint)
        :param n: number of values (default 1)
        :returns: cint (if n=1) or list of cint
        """
        res = [cls() for i in range(n)]
        readsocketc(client_id, *res)
        if n == 1:
            return res[0]
        else:
            return res

    @vectorized_classmethod
    def write_to_socket(self, client_id, values, message_type=ClientMessageType.NoType):
        """ Send a list of clear values to a client.

        :param client_id: Client id (regint)
        :param values: list of cint
        """
        writesocketc(client_id, message_type, *values)

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._load_mem(address, ldmc, ldmci)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self._store_in_mem(address, stmc, stmci)

    @staticmethod
    def in_immediate_range(value):
        return value < 2**31 and value >= -2**31

    @vectorize_init
    def __init__(self, val=None, size=None):
        """
        :param val: initialization (cint/regint/int/cgf2n or list thereof)
        :param size: vector size (int), defaults to 1 or size of list
        """
        super(cint, self).__init__('c', val=val, size=size)

    @vectorize
    def load_int(self, val):
        if val:
            # +1 for sign
            bit_length = 1 + int(math.ceil(math.log(abs(val))))
            if program.options.ring:
                assert(bit_length <= int(program.options.ring))
            elif program.options.field:
                program.curr_tape.require_bit_length(bit_length)
        if self.in_immediate_range(val):
            ldi(self, val)
        else:
            max = 2**31 - 1
            sign = abs(val) // val
            val = abs(val)
            chunks = []
            while val:
                mod = val % max
                val = (val - mod) // max
                chunks.append(mod)
            sum = cint(sign * chunks.pop())
            for i,chunk in enumerate(reversed(chunks)):
                sum *= max
                if i == len(chunks) - 1:
                    addci(self, sum, sign * chunk)
                elif chunk:
                    sum += sign * chunk

    @vectorize
    def to_regint(self, n_bits=64, dest=None):
        """ Convert to regint.

        :param n_bits: bit length (int)
        :return: regint """
        dest = regint() if dest is None else dest
        convmodp(dest, self, bitlength=n_bits)
        return dest

    def __mod__(self, other):
        """ Clear modulo.

        :param other: cint/regint/int """
        return self.clear_op(other, modc, modci)

    def __rmod__(self, other):
        """ Clear modulo.

        :param other: cint/regint/int """
        return self.coerce_op(other, modc, True)

    def less_than(self, other, bit_length):
        """ Clear comparison for particular bit length.

        :param other: cint/regint/int
        :param bit_length: signed bit length of inputs
        :return: 0/1 (regint), undefined if inputs outside range """
        if bit_length <= 64:
            return self < other
        else:
            diff = self - other
            shifted = diff >> (bit_length - 1)
            res = regint(shifted & 1)
            return res

    def __lt__(self, other):
        """ Clear 64-bit comparison.

        :param other: cint/regint/int
        :return: 0/1 (regint) """
        if isinstance(other, (type(self),int)):
            return regint(self) < other
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (type(self),int)):
            return regint(self) > other
        else:
            return NotImplemented

    def __le__(self, other):
        return 1 - (self > other)

    def __ge__(self, other):
        return 1 - (self < other)

    for op in __gt__, __le__, __ge__:
        op.__doc__ = __lt__.__doc__
    del op

    @vectorize
    def __eq__(self, other):
        """ Clear equality test.

        :param other: cint/regint/int
        :return: 0/1 (regint) """
        if not isinstance(other, (_clear, int)):
            return NotImplemented
        res = 1
        remaining = program.bit_length
        while remaining > 0:
            if isinstance(other, cint):
                o = other.to_regint(min(remaining, 64))
            else:
                o = other % 2 ** 64
            res *= (self.to_regint(min(remaining, 64)) == o)
            self >>= 64
            other >>= 64
            remaining -= 64
        return res

    def __lshift__(self, other):
        """ Clear left shift.

        :param other: cint/regint/int """
        return self.clear_op(other, shlc, shlci)

    def __rshift__(self, other):
        """ Clear right shift.

        :param other: cint/regint/int """
        return self.clear_op(other, shrc, shrci)

    def __neg__(self):
        """ Clear negation. """
        return 0 - self

    def __abs__(self):
        """ Clear absolute. """
        return (self >= 0).if_else(self, -self)

    @vectorize
    def __invert__(self):
        """ Clear inversion using global bit length. """
        res = cint()
        notc(res, self, program.bit_length)
        return res

    def __rpow__(self, base):
        """ Clear power of two.

        :param other: 2 """
        if base == 2:
            return 1 << self
        else:
            return NotImplemented

    @vectorize
    def __rlshift__(self, other):
        """ Clear shift.

        :param other: cint/regint/int """
        return cint(other) << self

    @vectorize
    def __rrshift__(self, other):
        """ Clear shift.

        :param other: cint/regint/int """
        return cint(other) >> self

    @read_mem_value
    def mod2m(self, other, bit_length=None, signed=None):
        """ Clear modulo a power of two.

        :param other: cint/regint/int """
        return self % 2**other

    @read_mem_value
    def right_shift(self, other, bit_length=None):
        """ Clear shift.

        :param other: cint/regint/int """
        return self >> other

    @read_mem_value
    def greater_than(self, other, bit_length=None):
        return self > other

    @vectorize
    def bit_decompose(self, bit_length=None):
        """ Clear bit decomposition.

        :param bit_length: number of bits (default is global bit length)
        :return: list of cint """
        if bit_length == 0:
            return []
        bit_length = bit_length or program.bit_length
        return floatingpoint.bits(self, bit_length)

    def legendre(self):
        """ Clear Legendre symbol computation. """
        res = cint()
        legendrec(res, self)
        return res

    def digest(self, num_bytes):
        """ Clear hashing (libsodium default). """
        res = cint()
        digestc(res, self, num_bytes)
        return res

    def print_if(self, string):
        """ Output if value is non-zero.

        :param string: Python string """
        cond_print_str(self, string)

    def output_if(self, cond):
        cond_print_plain(cond, self, cint(0))


class cgf2n(_clear, _gf2n):
    """
    Clear :math:`\mathrm{GF}(2^n)` value. n is 40 or 128,
    depending on USE_GF2N_LONG compile-time variable.
    """
    __slots__ = []
    instruction_type = 'gf2n'
    reg_type = 'cg'

    @classmethod
    def bit_compose(cls, bits, step=None):
        """ Clear :math:`\mathrm{GF}(2^n)` bit composition.

        :param bits: list of cgf2n
        :param step: set every :py:obj:`step`-th bit in output (defaults to 1) """
        size = bits[0].size
        res = cls(size=size)
        vgbitcom(size, res, step or 1, *bits)
        return res

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._load_mem(address, gldmc, gldmci)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self._store_in_mem(address, gstmc, gstmci)

    @staticmethod
    def in_immediate_range(value):
        return value < 2**32 and value >= 0

    def __init__(self, val=None, size=None):
        """
        :param val: initialization (cgf2n/cint/regint/int or list thereof)
        :param size: vector size (int), defaults to 1 or size of list
        """
        super(cgf2n, self).__init__('cg', val=val, size=size)

    @vectorize
    def load_int(self, val):
        if val < 0:
            raise CompilerError('Negative GF2n immediate')
        if self.in_immediate_range(val):
            gldi(self, val)
        else:
            chunks = []
            while val:
                mod = val % 2**32
                val >>= 32
                chunks.append(mod)
            sum = cgf2n(chunks.pop())
            for i,chunk in enumerate(reversed(chunks)):
                sum <<= 32
                if i == len(chunks) - 1:
                    gaddci(self, sum, chunk)
                elif chunk:
                    sum += chunk

    def __neg__(self):
        """ Identity. """
        return self

    @vectorize
    def __invert__(self):
        """ Clear bit-wise inversion. """
        res = cgf2n()
        gnotc(res, self)
        return res

    @vectorize
    def __lshift__(self, other):
        """ Left shift.

        :param other: compile-time (int) """
        if isinstance(other, int):
            res = cgf2n()
            gshlci(res, self, other)
            return res
        else:
            return NotImplemented

    @vectorize
    def __rshift__(self, other):
        """ Right shift.

        :param other: compile-time (int) """
        if isinstance(other, int):
            res = cgf2n()
            gshrci(res, self, other)
            return res
        else:
            return NotImplemented

    @vectorize
    def bit_decompose(self, bit_length=None, step=None):
        """ Clear bit decomposition.

        :param bit_length: number of bits (defaults to global :math:`\mathrm{GF}(2^n)` bit length)
        :param step: extract every :py:obj:`step`-th bit (defaults to 1) """
        bit_length = bit_length or program.galois_length
        step = step or 1
        res = [type(self)() for _ in range(bit_length // step)]
        gbitdec(self, step, *res)
        return res

class regint(_register, _int):
    """
    Clear 64-bit integer.
    Unlike :py:class:`cint` this is always a 64-bit integer.
    """
    __slots__ = []
    reg_type = 'ci'
    instruction_type = 'modp'
    mov = staticmethod(movint)

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._load_mem(address, ldmint, ldminti)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self._store_in_mem(address, stmint, stminti)

    @vectorized_classmethod
    def pop(cls):
        """ Pop from stack. """
        res = cls()
        popint(res)
        return res

    @vectorized_classmethod
    def push(cls, value):
        """ Push to stack.

        :param value: any convertible type """
        pushint(cls.conv(value))

    @vectorized_classmethod
    def get_random(cls, bit_length):
        """ Public insecure randomness.

        :param bit_length: number of bits (int)
        :param size: vector size (int, default 1)
        """
        if isinstance(bit_length, int):
            bit_length = regint(bit_length)
        res = cls()
        rand(res, bit_length)
        return res

    @classmethod
    def inc(cls, size, base=0, step=1, repeat=1, wrap=None):
        """
        Produce :py:class:`regint` vector with certain patterns.
        This is particularly useful for :py:meth:`SubMultiArray.direct_mul`.

        :param size: Result size
        :param base: First value
        :param step: Increase step
        :param repeat: Repeate this many times
        :param wrap: Start over after this many increases

        The following produces (1, 1, 1, 3, 3, 3, 5, 5, 5, 7)::

            regint.inc(10, 1, 2, 3)

        """
        res = regint(size=size)
        if wrap is None:
            wrap = size
        incint(res, cls.conv(base, size=1), step, repeat, wrap)
        return res

    @vectorized_classmethod
    def read_from_socket(cls, client_id, n=1):
        """ Receive clear integer value(s) from client.

        :param client_id: Client id (regint)
        :param n: number of values (default 1)
        :returns: regint (if n=1) or list of regint
        """
        res = [cls() for i in range(n)]
        readsocketint(client_id, *res)
        if n == 1:
            return res[0]
        else:
            return res

    @vectorized_classmethod
    def write_to_socket(self, client_id, values, message_type=ClientMessageType.NoType):
        """ Send a list of clear integers to a client.

        :param client_id: Client id (regint)
        :param values: list of regint
        """
        writesocketint(client_id, message_type, *values)

    @vectorize_init
    def __init__(self, val=None, size=None):
        """
        :param val: initialization (cint/cgf2n/regint/int or list thereof)
        :param size: vector size (int), defaults to 1 or size of list
        """
        super(regint, self).__init__(self.reg_type, val=val, size=size)

    def load_int(self, val):
        if cint.in_immediate_range(val):
            ldint(self, val)
        else:
            lower = val % 2**32
            upper = val >> 32
            if lower >= 2**31:
                lower -= 2**32
                upper += 1
            addint(self, regint(upper) * regint(2**16)**2, regint(lower))

    @read_mem_value
    def load_other(self, val):
        if isinstance(val, cgf2n):
            gconvgf2n(self, val)
        elif isinstance(val, regint):
            addint(self, val, regint(0))
        else:
            try:
                val.to_regint(dest=self)
            except AttributeError:
                raise CompilerError("Cannot convert '%s' to integer" % \
                                    type(val))

    @vectorize
    @read_mem_value
    def int_op(self, other, inst, reverse=False):
        try:
            other = self.conv(other)
        except:
            return NotImplemented
        res = regint()
        if reverse:
            inst(res, other, self)
        else:
            inst(res, self, other)
        return res

    def add(self, other):
        """ Clear addition.

        :param other: regint/cint/int """
        return self.int_op(other, addint)

    def __sub__(self, other):
        """ Clear subtraction.

        :param other: regint/cint/int """
        return self.int_op(other, subint)

    def __rsub__(self, other):
        return self.int_op(other, subint, True)
    __rsub__.__doc__ = __sub__.__doc__

    def mul(self, other):
        """ Clear multiplication.

        :param other: regint/cint/int """
        return self.int_op(other, mulint)

    def __neg__(self):
        """ Clear negation. """
        return 0 - self

    def __floordiv__(self, other):
        """ Clear integer division (rounding to floor).

        :param other: regint/cint/int """
        return self.int_op(other, divint)

    def __rfloordiv__(self, other):
        return self.int_op(other, divint, True)
    __rfloordiv__.__doc__ = __floordiv__.__doc__

    __truediv__ = __floordiv__
    __rtruediv__ = __rfloordiv__

    def __mod__(self, other):
        """ Clear modulo computation.

        :param other: regint/cint/int """
        return self - (self / other) * other

    def __rmod__(self, other):
        """ Clear modulo computation.

        :param other: regint/cint/int """
        return regint(other) % self

    def __rpow__(self, other):
        """ Clear power of two computation.

        :param other: regint/cint/int
        :rtype: cint """
        return other**cint(self)

    def __eq__(self, other):
        """ Clear comparison.

        :param other: regint/cint/int
        :return: 0/1 """
        return self.int_op(other, eqc)

    def __ne__(self, other):
        return 1 - (self == other)

    def __lt__(self, other):
        return self.int_op(other, ltc)

    def __gt__(self, other):
        return self.int_op(other, gtc)

    def __le__(self, other):
        return 1 - (self > other)

    def __ge__(self, other):
        return 1 - (self < other)

    for op in __le__, __lt__, __ge__, __gt__, __ne__:
        op.__doc__ = __eq__.__doc__
    del op

    def __lshift__(self, other):
        """ Clear shift.

        :param other: regint/cint/int """
        if isinstance(other, int):
            return self * 2**other
        else:
            return regint(cint(self) << other)

    def __rshift__(self, other):
        if isinstance(other, int):
            return self / 2**other
        else:
            return regint(cint(self) >> other)

    def __rlshift__(self, other):
        return regint(other << cint(self))

    def __rrshift__(self, other):
        return regint(other >> cint(self))

    for op in __rshift__, __rlshift__, __rrshift__:
        op.__doc__ = __lshift__.__doc__
    del op

    def __and__(self, other):
        """ Clear bit-wise AND.

        :param other: regint/cint/int """
        return regint(other & cint(self))

    def __or__(self, other):
        """ Clear bit-wise OR.

        :param other: regint/cint/int """
        return regint(other | cint(self))

    def __xor__(self, other):
        """ Clear bit-wise XOR.

        :param other: regint/cint/int """
        return regint(other ^ cint(self))

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def mod2m(self, *args, **kwargs):
        """ Clear modulo a power of two.

        :rtype: cint """
        return cint(self).mod2m(*args, **kwargs)

    @vectorize
    def bit_decompose(self, bit_length=None):
        """ Clear bit decomposition.

        :param bit_length: number of bits (defaults to global bit length)
        :return: list of regint """
        bit_length = bit_length or min(64, program.bit_length)
        if bit_length > 64:
            raise CompilerError('too many bits demanded')
        res = [regint() for i in range(bit_length)]
        bitdecint(self, *res)
        return res

    @staticmethod
    def bit_compose(bits):
        """ Clear bit composition.

        :param bits: list of regint/cint/int """
        two = regint(2)
        res = 0
        for bit in reversed(bits):
            res *= two
            res += bit
        return res

    def shuffle(self):
        """ Returns insecure shuffle of vector. """
        res = regint(size=len(self))
        shuffle(res, self)
        return res

    def reveal(self):
        """ Identity. """
        return self

    def print_reg_plain(self):
        """ Output. """
        print_int(self)

    def print_if(self, string):
        """ Output string if value is non-zero.

        :param string: Python string """
        cint(self).print_if(string)

    def output_if(self, cond):
        cint(self).output_if(cond)

class localint(object):
    """ Local integer that must prevented from leaking into the secure
    computation. Uses regint internally. """

    def __init__(self, value=None):
        """ :param value: initialization, convertible to regint """
        self._v = regint(value)
        self.size = 1

    def output(self):
        """ Output. """
        self._v.print_reg_plain()

    """ Local comparison. """
    __lt__ = lambda self, other: localint(self._v < other)
    __le__ = lambda self, other: localint(self._v <= other)
    __gt__ = lambda self, other: localint(self._v > other)
    __ge__ = lambda self, other: localint(self._v >= other)
    __eq__ = lambda self, other: localint(self._v == other)
    __ne__ = lambda self, other: localint(self._v != other)

class personal(object):
    def __init__(self, player, value):
        self.player = player
        self._v = value

class longint:
    def __init__(self, value, length=None, n_limbs=None):
        assert length is None or n_limbs is None
        if isinstance(value, longint):
            if n_limbs is None:
                n_limbs = int(math.ceil(length / 64))
            assert n_limbs <= len(value.v)
            self.v = value.v[:n_limbs]
        elif isinstance(value, list):
            assert length is None
            self.v = value[:]
        else:
            if length is None:
                length = 64 * n_limbs
            if isinstance(value, int):
                self.v = [(value >> i) for i in range(0, length, 64)]
            else:
                self.v = [(value >> i).to_regint(0)
                          for i in range(0, length, 64)]

    def coerce(self, other):
        return longint(other, n_limbs=len(self.v))

    def __eq__(self, other):
        return reduce(operator.mul, (x == y for x, y in
                                     zip(self.v, self.coerce(other).v)))

    def __add__(self, other):
        other = self.coerce(other)
        assert len(self.v) == len(other.v)
        res = []
        carry = 0
        for x, y in zip(self.v, other.v):
            res.append(x + y + carry)
            carry = util.if_else(carry, (res[-1] + 2 ** 63) <= (x + 2 ** 63),
                                 (res[-1] + 2 ** 63) < (x + 2 ** 63))
        return longint(res)

    __radd__ = __add__

    def __sub__(self, other):
        return self + -other

    def bit_decompose(self, bit_length):
        assert bit_length <= 64 * len(self.v)
        res = []
        for x in self.v:
            res += x.bit_decompose(64)
        return res[:bit_length]

class _secret(_register):
    __slots__ = []

    mov = staticmethod(set_instruction_type(movs))
    PreOR = staticmethod(lambda l: floatingpoint.PreORC(l))
    PreOp = staticmethod(lambda op, l: floatingpoint.PreOpL(op, l))

    @vectorized_classmethod
    @set_instruction_type
    def get_input_from(cls, player):
        """ Secret input from player.

        :param player: public (regint/cint/int)
        :param size: vector size (int, default 1)
        """
        res = cls()
        asm_input(res, player)
        return res

    @vectorized_classmethod
    @set_instruction_type
    def get_random_triple(cls):
        """ Secret random triple according to security model.

        :return: :math:`(a, b, ab)`
        :param size: vector size (int, default 1)
        """
        res = (cls(), cls(), cls())
        triple(*res)
        return res

    @vectorized_classmethod
    @set_instruction_type
    def get_random_bit(cls):
        """ Secret random bit according to security model.

        :return: 0/1 50-50
        :param size: vector size (int, default 1)
        """
        res = cls()
        bit(res)
        return res

    @vectorized_classmethod
    @set_instruction_type
    def get_random_square(cls):
        """ Secret random square according to security model.

        :return: :math:`(a, a^2)`
        :param size: vector size (int, default 1)
        """
        res = (cls(), cls())
        square(*res)
        return res

    @vectorized_classmethod
    @set_instruction_type
    def get_random_inverse(cls):
        """ Secret random inverse tuple according to security model.

        :return: :math:`(a, a^{-1})`
        :param size: vector size (int, default 1)
        """
        res = (cls(), cls())
        inverse(*res)
        return res

    @vectorized_classmethod
    @set_instruction_type
    def get_random_input_mask_for(cls, player):
        res = cls()
        inputmask(res, player)
        return res

    @classmethod
    @set_instruction_type
    def dot_product(cls, x, y):
        """
        Secret dot product.

        :param x: Iterable of secret values
        :param y: Iterable of secret values of same length and type

        :rtype: same as inputs
        """
        x = list(x)
        set_global_vector_size(x[0].size)
        res = cls()
        dotprods(res, x, y)
        reset_global_vector_size()
        return res

    @classmethod
    @set_instruction_type
    def row_matrix_mul(cls, row, matrix, res_params=None):
        assert len(row) == len(matrix)
        size = len(matrix[0])
        res = cls(size=size)
        dotprods(*sum(([res[j], row, [matrix[k][j] for k in range(len(row))]]
                       for j in range(size)), []))
        return res

    @classmethod
    @set_instruction_type
    def matrix_mul(cls, A, B, n, res_params=None):
        assert len(A) % n == 0
        assert len(B) % n == 0
        size = len(A) * len(B) // n**2
        res = cls(size=size)
        n_rows = len(A) // n
        n_cols = len(B) // n
        matmuls(res, A, B, n_rows, n, n_cols)
        return res

    @no_doc
    def __init__(self, reg_type, val=None, size=None):
        if isinstance(val, self.clear_type):
            size = val.size
        super(_secret, self).__init__(reg_type, val=val, size=size)

    @set_instruction_type
    @vectorize
    def load_int(self, val):
        if self.clear_type.in_immediate_range(val):
            ldsi(self, val)
        else:
            self.load_clear(self.clear_type(val))

    @vectorize
    def load_clear(self, val):
        addm(self, self.__class__(0), val)

    @set_instruction_type
    @read_mem_value
    @vectorize
    def load_other(self, val):
        from Compiler.GC.types import sbits, sbitvec
        if isinstance(val, self.clear_type):
            self.load_clear(val)
        elif isinstance(val, type(self)):
            movs(self, val)
        elif isinstance(val, sbits):
            assert(val.n == self.size)
            r = self.get_dabit()
            movs(self, r[0].bit_xor((r[1] ^ val).reveal().to_regint_by_bit()))
        elif isinstance(val, sbitvec):
            assert(sum(x.n for x in val.v) == self.size)
            for val_part, base in zip(val, range(0, self.size, 64)):
                left = min(64, self.size - base)
                r = self.get_dabit(size=left)
                v = regint(size=left)
                bitdecint_class(regint((r[1] ^ val_part).reveal()), *v)
                part = r[0].bit_xor(v)
                vmovs(left, self.get_vector(base, left), part)
        else:
            self.load_clear(self.clear_type(val))

    @classmethod
    def bit_compose(cls, bits):
        """ Compose value from bits.

        :param bits: iterable of any type convertible to sint """
        from Compiler.GC.types import sbits, sbitintvec
        bits = list(bits)
        if (program.use_edabit() or program.use_split()) and isinstance(bits[0], sbits):
            if program.use_edabit():
                mask = cls.get_edabit(len(bits), strict=True, size=bits[0].n)
            else:
                tmp = sint(size=bits[0].n)
                randoms(tmp, len(bits))
                n_overflow_bits = min(program.use_split().bit_length(),
                                      int(program.options.ring) - len(bits))
                mask_bits = tmp.bit_decompose(len(bits) + n_overflow_bits,
                                              maybe_mixed=True)
                if n_overflow_bits:
                    overflow = sint.bit_compose(
                        sint.conv(x) for x in mask_bits[-n_overflow_bits:])
                    mask = tmp - (overflow << len(bits)), \
                        mask_bits[:-n_overflow_bits]
                else:
                    mask = tmp, mask_bits
            t = sbitintvec.get_type(len(bits) + 1)
            masked = t.from_vec(mask[1] + [0]) + t.from_vec(bits + [0])
            overflow = masked.v[-1]
            masked = cls.bit_compose(x.reveal().to_regint_by_bit() for x in masked.v[:-1])
            return masked - mask[0] + (cls(overflow) << len(bits))
        else:
            return super(_secret, cls).bit_compose(bits)

    @set_instruction_type
    @read_mem_value
    @vectorize
    def secret_op(self, other, s_inst, m_inst, si_inst, reverse=False):
        res = self.prep_res(other)
        cls = type(res)
        if isinstance(other, regint):
            other = res.clear_type(other)
        if isinstance(other, cls):
            if reverse:
                s_inst(res, other, self)
            else:
                s_inst(res, self, other)
        elif isinstance(other, res.clear_type):
            if reverse:
                m_inst(res, other, self)
            else:
                m_inst(res, self, other)
        elif isinstance(other, int):
            if self.clear_type.in_immediate_range(other):
                si_inst(res, self, other)
            else:
                if reverse:
                    m_inst(res, res.clear_type(other), self)
                else:
                    m_inst(res, self, res.clear_type(other))
        else:
            return NotImplemented
        return res

    def add(self, other):
        """ Secret addition.

        :param other: any compatible type """
        return self.secret_op(other, adds, addm, addsi)

    @set_instruction_type
    def mul(self, other):
        """ Secret multiplication. Either both operands have the same
        size or one size 1 for a value-vector multiplication.

        :param other: any compatible type """
        if isinstance(other, _secret) and (1 in (self.size, other.size)) \
           and (self.size, other.size) != (1, 1):
            x, y = (other, self) if self.size < other.size else (self, other)
            res = type(self)(size=x.size)
            mulrs(res, x, y)
            return res
        return self.secret_op(other, muls, mulm, mulsi)

    def __sub__(self, other):
        """ Secret subtraction.

        :param other: any compatible type """
        return self.secret_op(other, subs, subml, subsi)

    def __rsub__(self, other):
        return self.secret_op(other, subs, submr, subsfi, True)
    __rsub__.__doc__ = __sub__.__doc__

    @vectorize
    def __truediv__(self, other):
        """ Secret field division.

        :param other: any compatible type """
        return self * (self.clear_type(1) / other)

    @vectorize
    def __rtruediv__(self, other):
        a,b = self.get_random_inverse()
        return other * a / (a * self).reveal()
    __rtruediv__.__doc__ = __truediv__.__doc__

    @set_instruction_type
    @vectorize
    def square(self):
        """ Secret square. """
        if program.use_square():
            res = self.__class__()
            sqrs(res, self)
            return res
        else:
            return self * self

    @set_instruction_type
    @vectorize
    def reveal(self):
        """ Reveal secret value publicly.

        :rtype: relevant clear type """
        res = self.clear_type()
        asm_open(res, self)
        return res

    @set_instruction_type
    def reveal_to(self, player):
        """ Reveal secret value to :py:obj:`player`.
        Result written to ``Player-Data/Private-Output-P<player>``

        :param player: int
        :returns: value to be used with :py:func:`~Compiler.library.print_ln_to`
        """
        masked = self.__class__()
        res = personal(player, self.clear_type())
        startprivateoutput(masked, self, player)
        stopprivateoutput(res._v, masked.reveal(), player)
        return res


class sint(_secret, _int):
    """
    Secret integer in the protocol-specific domain.

    Most non-linear operations require compile-time parameters for bit
    length and statistical security. They default to the global
    parameters set by :py:meth:`program.set_bit_length` and
    :py:meth:`program.set_security`. The acceptable minimum for statistical
    security is considered to be 40.  The defaults for the parameters
    is output at the beginning of the compilation.

    If the computation domain is modulo a power of two, the
    operands will be truncated to the bit length, and the security
    parameter does not matter. Modulo prime, the behaviour is
    undefined and potentially insecure if the operands are longer than
    the bit length.
    """
    __slots__ = []
    instruction_type = 'modp'
    clear_type = cint
    reg_type = 's'

    PreOp = staticmethod(floatingpoint.PreOpL)
    PreOR = staticmethod(floatingpoint.PreOR)
    get_type = staticmethod(lambda n: sint)

    @staticmethod
    def require_bit_length(n_bits):
        if program.options.ring:
            if int(program.options.ring) < n_bits:
                raise CompilerError('computation modulus too small')
        else:
            program.curr_tape.require_bit_length(n_bits)

    @vectorized_classmethod
    def get_random_int(cls, bits):
        """ Secret random n-bit number according to security model.

        :param bits: compile-time integer (int)
        :param size: vector size (int, default 1)
        """
        if program.use_edabit():
            return sint.get_edabit(bits, True)[0]
        elif program.use_split() > 2 and program.use_split() < 5:
            tmp = sint()
            randoms(tmp, bits)
            x = tmp.split_to_two_summands(bits, True)
            carry = comparison.CarryOutRawLE(x[1][:bits], x[0][:bits])
            if program.use_split() > 3:
                from .GC.types import sbitint
                x = sbitint.full_adder(carry, x[0][bits], x[1][bits])
                overflow = sint.conv(x[1]) * 2 + sint.conv(x[0])
            else:
                overflow = sint.conv(carry) + sint.conv(x[0][bits])
            return tmp - (overflow << bits)
        res = sint()
        comparison.PRandInt(res, bits)
        return res

    @vectorized_classmethod
    def get_random(cls):
        """ Secret random ring element according to security model.

        :param size: vector size (int, default 1)
        """
        res = sint()
        randomfulls(res)
        return res

    @vectorized_classmethod
    def get_input_from(cls, player):
        """ Secret input.

        :param player: public (regint/cint/int)
        :param size: vector size (int, default 1)
        """
        res = cls()
        inputmixed('int', res, player)
        return res

    @vectorized_classmethod
    def get_dabit(cls):
        """ Bit in arithmetic and binary circuit according to security model """
        from Compiler.GC.types import sbits
        res = cls(), sbits.get_type(get_global_vector_size())()
        dabit(*res)
        return res

    @vectorized_classmethod
    def get_edabit(cls, n_bits, strict=False):
        """ Bits in arithmetic and binary circuit """
        """ according to security model """
        if not program.use_edabit_for(strict, n_bits):
            if program.use_dabit:
                a, b = zip(*(sint.get_dabit() for i in range(n_bits)))
                return sint.bit_compose(a), b
            else:
                a = [sint.get_random_bit() for i in range(n_bits)]
                return sint.bit_compose(a), a
        program.curr_tape.require_bit_length(n_bits - 1)
        whole = cls()
        size = get_global_vector_size()
        from Compiler.GC.types import sbits, sbitvec
        bits = [sbits.get_type(size)() for i in range(n_bits)]
        if strict:
            sedabit(whole, *bits)
        else:
            edabit(whole, *bits)
        return whole, bits

    @staticmethod
    def long_one():
        return 1

    @staticmethod
    @vectorize
    def bit_decompose_clear(a, n_bits):
        return floatingpoint.bits(a, n_bits)

    @vectorized_classmethod
    def get_raw_input_from(cls, player):
        res = cls()
        rawinput(player, res)
        return res

    @classmethod
    def receive_from_client(cls, n, client_id, message_type=ClientMessageType.NoType):
        """ Securely obtain shares of values input by a client.

        :param n: number of inputs (int)
        :param client_id: regint
        """
        # send shares of a triple to client
        triples = list(itertools.chain(*(sint.get_random_triple() for i in range(n))))
        sint.write_shares_to_socket(client_id, triples, message_type)

        received = cint.read_from_socket(client_id, n)
        y = [0] * n
        for i in range(n):
            y[i] = received[i] - triples[i * 3]
        return y

    @vectorized_classmethod
    def read_from_socket(cls, client_id, n=1):
        """ Receive secret-shared value(s) from client.

        :param client_id: Client id (regint)
        :param n: number of values (default 1)
        :returns: sint (if n=1) or list of sint
        """
        res = [cls() for i in range(n)]
        readsockets(client_id, *res)
        if n == 1:
            return res[0]
        else:
            return res

    @vectorize
    def write_share_to_socket(self, client_id, message_type=ClientMessageType.NoType):
        """ Send only share to socket """
        writesocketshare(client_id, message_type, self)

    @vectorized_classmethod
    def write_shares_to_socket(cls, client_id, values,
                               message_type=ClientMessageType.NoType):
        """ Send shares of a list of values to a specified client socket.

        :param client_id: regint
        :param values: list of sint
        """
        writesocketshare(client_id, message_type, *values)

    @classmethod
    def read_from_file(cls, start, n_items):
        """ Read shares from ``Persistence/Transactions-P<playerno>.data``.

        :param start: starting position in number of shares from beginning (int/regint/cint)
        :param n_items: number of items (int)
        :returns: destination for final position, -1 for eof reached, or -2 for file not found (regint)
        :returns: list of shares
        """
        shares = [cls(size=1) for i in range(n_items)]
        stop = regint()
        readsharesfromfile(regint.conv(start), stop, *shares)
        return stop, shares

    @staticmethod
    def write_to_file(shares):
        """ Write shares to ``Persistence/Transactions-P<playerno>.data``
        (appending at the end).

        :param: shares (list or iterable of sint)
        """
        for share in shares:
            assert isinstance(share, sint)
            assert share.size == 1
        writesharestofile(*shares)

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._load_mem(address, ldms, ldmsi)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self._store_in_mem(address, stms, stmsi)

    @classmethod
    def direct_matrix_mul(cls, A, B, n, m, l, reduce=None, indices=None):
        if indices is None:
            indices = [regint.inc(i) for i in (n, m, m, l)]
        res = cls(size=indices[0].size * indices[3].size)
        matmulsm(res, regint(A), regint(B), len(indices[0]), len(indices[1]),
                 len(indices[3]), *(list(indices) + [m, l]))
        return res

    def __init__(self, val=None, size=None):
        """
        :param val: initialization (sint/cint/regint/int/cgf2n or list thereof)
        :param size: vector size (int), defaults to 1 or size of list
        """
        super(sint, self).__init__('s', val=val, size=size)

    @vectorize
    def __neg__(self):
        """ Secret negation. """
        return 0 - self

    @vectorize
    def __abs__(self):
        """ Secret absolute. Uses global parameters for comparison. """
        return (self >= 0).if_else(self, -self)

    @read_mem_value
    @vectorize
    def __lt__(self, other, bit_length=None, security=None):
        """ Secret comparison (signed).

        :param other: sint/cint/regint/int
        :return: 0/1 (sint) """
        res = sintbit()
        comparison.LTZ(res, self - other,
                       (bit_length or program.bit_length) + 1,
                       security or program.security)
        return res

    @read_mem_value
    @vectorize
    def __gt__(self, other, bit_length=None, security=None):
        res = sintbit()
        comparison.LTZ(res, other - self,
                       (bit_length or program.bit_length) + 1,
                       security or program.security)
        return res

    def __le__(self, other, bit_length=None, security=None):
        return 1 - self.greater_than(other, bit_length, security)

    def __ge__(self, other, bit_length=None, security=None):
        return 1 - self.less_than(other, bit_length, security)

    @read_mem_value
    @vectorize
    def __eq__(self, other, bit_length=None, security=None):
        return floatingpoint.EQZ(self - other, bit_length or program.bit_length,
                                 security or program.security)

    def __ne__(self, other, bit_length=None, security=None):
        return 1 - self.equal(other, bit_length, security)

    less_than = __lt__
    greater_than = __gt__
    less_equal = __le__
    greater_equal = __ge__
    equal = __eq__
    not_equal = __ne__

    for op in __gt__, __le__, __ge__, __eq__, __ne__:
        op.__doc__ = __lt__.__doc__
    del op

    @vectorize
    def __mod__(self, modulus):
        """ Secret modulo computation.
        Uses global parameters for bit length and security.

        :param modulus: power of two (int) """
        if isinstance(modulus, int):
            l = math.log(modulus, 2)
            if 2**int(round(l)) == modulus:
                return self.mod2m(int(l))
        raise NotImplementedError('Modulo only implemented for powers of two.')

    @vectorize
    @read_mem_value
    def mod2m(self, m, bit_length=None, security=None, signed=True):
        """ Secret modulo power of two.

        :param m: secret or public integer (sint/cint/regint/int) """
        bit_length = bit_length or program.bit_length
        security = security or program.security
        if isinstance(m, int):
            if m == 0:
                return 0
            if m >= bit_length:
                return self
            res = sint()
            comparison.Mod2m(res, self, bit_length, m, security, signed)
        else:
            res, pow2 = floatingpoint.Trunc(self, bit_length, m, security, True)
        return res

    @vectorize
    def __rpow__(self, base):
        """ Secret power computation. Base must be two.
        Uses global parameters for bit length and security. """
        if base == 2:
            return self.pow2()
        else:
            return NotImplemented

    @vectorize
    def pow2(self, bit_length=None, security=None):
        """ Secret power of two. """
        return floatingpoint.Pow2(self, bit_length or program.bit_length, \
                                      security or program.security)

    def __lshift__(self, other, bit_length=None, security=None):
        """ Secret left shift.

        :param other: secret or public integer (sint/cint/regint/int) """
        return self * util.pow2_value(other, bit_length, security)

    @vectorize
    @read_mem_value
    def __rshift__(self, other, bit_length=None, security=None, signed=True):
        """ Secret right shift.

        :param other: secret or public integer (sint/cint/regint/int) """
        bit_length = bit_length or program.bit_length
        security = security or program.security
        if isinstance(other, int):
            if other == 0:
                return self
            res = sint()
            comparison.Trunc(res, self, bit_length, other, security, signed)
            return res
        elif isinstance(other, sint):
            return floatingpoint.Trunc(self, bit_length, other, security)
        else:
            return floatingpoint.Trunc(self, bit_length, sint(other), security)

    left_shift = __lshift__
    right_shift = __rshift__

    def __rlshift__(self, other):
        """ Secret left shift.
        Bit length of :py:obj:`self` uses global value.

        :param other: secret or public integer (sint/cint/regint/int) """
        return other * 2**self

    @vectorize
    def __rrshift__(self, other):
        """ Secret right shift.

        :param other: secret or public integer (sint/cint/regint/int) of globale bit length if secret """
        return floatingpoint.Trunc(other, program.bit_length, self, program.security)

    @vectorize
    def bit_decompose(self, bit_length=None, security=None, maybe_mixed=False):
        """ Secret bit decomposition. """
        if bit_length == 0:
            return []
        bit_length = bit_length or program.bit_length
        assert program.security == security or program.security
        return program.non_linear.bit_dec(self, bit_length, bit_length,
                                          maybe_mixed)

    def TruncMul(self, other, k, m, kappa=None, nearest=False):
        return (self * other).round(k, m, kappa, nearest, signed=True)

    def TruncPr(self, k, m, kappa=None, signed=True):
        return floatingpoint.TruncPr(self, k, m, kappa, signed=signed)

    @vectorize
    def round(self, k, m, kappa=None, nearest=False, signed=False):
        """ Truncate and maybe round secret :py:obj:`k`-bit integer
        by :py:obj:`m` bits. :py:obj:`m` can be secret if
        :py:obj:`nearest` is false, in which case the truncation will be
        exact. For public :py:obj:`m`, :py:obj:`nearest` chooses
        between nearest rounding (rounding half up) and probablistic
        truncation.

        :param k: int
        :param m: secret or compile-time integer (sint/int)
        :param kappa: statistical security parameter (int)
        :param nearest: bool
        :param signed: bool """
        kappa = kappa or program.security
        secret = isinstance(m, sint)
        if nearest:
            if secret:
                raise NotImplementedError()
            return comparison.TruncRoundNearest(self, k, m, kappa,
                                                signed=signed)
        else:
            if secret:
                return floatingpoint.Trunc(self, k, m, kappa)
            return self.TruncPr(k, m, kappa, signed=signed)

    def Norm(self, k, f, kappa=None, simplex_flag=False):
        return library.Norm(self, k, f, kappa, simplex_flag)

    @vectorize
    def int_div(self, other, bit_length=None, security=None):
        """ Secret integer division.

        :param other: sint """
        k = bit_length or program.bit_length
        kappa = security or program.security
        tmp = library.IntDiv(self, other, k, kappa)
        res = type(self)()
        comparison.Trunc(res, tmp, 2 * k, k, kappa, True)
        return res

    def trunc_zeros(self, n_zeros, bit_length=None, signed=True):
        bit_length = bit_length or program.bit_length
        return comparison.TruncZeros(self, bit_length, n_zeros, signed)

    @staticmethod
    def two_power(n):
        return floatingpoint.two_power(n)

    def split_to_n_summands(self, length, n):
        comparison.require_ring_size(length, 'splitting')
        from .GC.types import sbits
        from .GC.instructions import split
        columns = [[sbits.get_type(self.size)()
                    for i in range(n)] for i in range(length)]
        split(n, self, *sum(columns, []))
        return columns

    def split_to_two_summands(self, length, get_carry=False):
        n = program.use_split()
        assert n
        columns = self.split_to_n_summands(length, n)
        return _bitint.wallace_tree_without_finish(columns, get_carry)

    @vectorize
    def raw_right_shift(self, length):
        res = sint()
        shrsi(res, self, length)
        return res

    def raw_mod2m(self, m):
        return self - (self.raw_right_shift(m) << m)

    @vectorize
    def reveal_to(self, player):
        """ Reveal secret value to :py:obj:`player`.
        Result potentially written to
        ``Player-Data/Private-Output-P<player>``, but not if
        :py:obj:`player` is a :py:class:`regint`.

        :param player: public integer (int/regint/cint):
        :returns: value to be used with :py:func:`~Compiler.library.print_ln_to`
        """
        if not util.is_constant(player) or self.size > 1:
            secret_mask = sint()
            player_mask = cint()
            inputmaskreg(secret_mask, player_mask, player)
            return personal(player,
                            (self + secret_mask).reveal() - player_mask)
        else:
            return super(sint, self).reveal_to(player)

class sintbit(sint):
    @classmethod
    def prep_res(cls, other):
        return sint()

    def load_other(self, other):
        if isinstance(other, sint):
            movs(self, other)
        else:
            super(sintbit, self).load_other(other)

    @vectorize
    def __and__(self, other):
        if isinstance(other, sintbit):
            res = sintbit()
            muls(res, self, other)
            return res
        elif util.is_zero(other):
            return 0
        elif util.is_one(other):
            return self
        else:
            return NotImplemented

    @vectorize
    def __or__(self, other):
        if isinstance(other, sintbit):
            res = sintbit()
            adds(res, self, other - self * other)
            return res
        elif util.is_zero(other):
            return self
        elif util.is_one(other):
            return 1
        else:
            return NotImplemented

    @vectorize
    def __xor__(self, other):
        if isinstance(other, sintbit):
            res = sintbit()
            adds(res, self, other - 2 * self * other)
            return res
        elif util.is_zero(other):
            return self
        elif util.is_one(other):
            return 1
        else:
            return NotImplemented

    @vectorize
    def __rsub__(self, other):
        if util.is_one(other):
            res = sintbit()
            subsfi(res, self, 1)
            return res
        else:
            return super(sintbit, self).__rsub__(other)

class sgf2n(_secret, _gf2n):
    """ Secret :math:`\mathrm{GF}(2^n)` value. """
    __slots__ = []
    instruction_type = 'gf2n'
    clear_type = cgf2n
    reg_type = 'sg'

    @classmethod
    def get_type(cls, length):
        return cls

    @classmethod
    def get_raw_input_from(cls, player):
        res = cls()
        grawinput(player, res)
        return res

    def add(self, other):
        """ Secret :math:`\mathrm{GF}(2^n)` addition (XOR).

        :param other: sg2fn/cgf2n/regint/int """
        if isinstance(other, sgf2nint):
            return NotImplemented
        else:
            return super(sgf2n, self).add(other)

    def mul(self, other):
        """ Secret :math:`\mathrm{GF}(2^n)` multiplication.

        :param other: sg2fn/cgf2n/regint/int """
        if isinstance(other, (sgf2nint)):
            return NotImplemented
        else:
            return super(sgf2n, self).mul(other)

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._load_mem(address, gldms, gldmsi)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self._store_in_mem(address, gstms, gstmsi)

    def __init__(self, val=None, size=None):
        """
        :param val: initialization (sgf2n/cgf2n/regint/int/cint or list thereof)
        :param size: vector size (int), defaults to 1 or size of list
        """
        super(sgf2n, self).__init__('sg', val=val, size=size)

    def __neg__(self):
        """ Identity. """
        return self

    @vectorize
    def __invert__(self):
        """ Secret bit-wise inversion. """
        return self ^ cgf2n(2**program.galois_length - 1)

    def __xor__(self, other):
        """ Secret bit-wise XOR.

        :param other: sg2fn/cgf2n/regint/int """
        if is_zero(other):
            return self
        else:
            return super(sgf2n, self).add(other)

    __rxor__ = __xor__

    @vectorize
    def __and__(self, other):
        """ Secret bit-wise AND.

        :param other: sg2fn/cgf2n/regint/int """
        if isinstance(other, int):
            other_bits = [(other >> i) & 1 \
                              for i in range(program.galois_length)]
        else:
            other_bits = other.bit_decompose()
        self_bits = self.bit_decompose()
        return sum((x * y) << i \
                       for i,(x,y) in enumerate(zip(self_bits, other_bits)))

    __rand__ = __and__

    @vectorize
    def __lshift__(self, other):
        """ Secret left shift py public value.

        :param other: regint/cint/int """
        return self * cgf2n(1 << other)

    @vectorize
    def right_shift(self, other, bit_length=None):
        """ Secret right shift by public value:

        :param other: compile-time (int)
        :param bit_length: number of bits of :py:obj:`self` (defaults to :math:`\mathrm{GF}(2^n)` bit length) """
        bits = self.bit_decompose(bit_length)
        return sum(b << i for i,b in enumerate(bits[other:]))

    def equal(self, other, bit_length=None, expand=1):
        """ Secret comparison.

        :param other: sgf2n/cgf2n/regint/int
        :return: 0/1 (sgf2n) """
        bits = [1 - bit for bit in (self - other).bit_decompose(bit_length)][::expand]
        while len(bits) > 1:
            bits.insert(0, bits.pop() * bits.pop())
        return bits[0]

    def not_equal(self, other, bit_length=None):
        """ Secret comparison. """
        return 1 - self.equal(other, bit_length)
    not_equal.__doc__ = equal.__doc__

    __eq__ = equal
    __ne__ = not_equal

    @vectorize
    def bit_decompose(self, bit_length=None, step=1):
        """ Secret bit decomposition.

        :param bit_length: number of bits
        :param step: use every :py:obj:`step`-th bit
        :return: list of sgf2n """
        if bit_length == 0:
            return []
        bit_length = bit_length or program.galois_length
        random_bits = [self.get_random_bit() \
                           for i in range(0, bit_length, step)]

        one = cgf2n(1)
        masked = sum([b * (one << (i * step)) for i,b in enumerate(random_bits)], self).reveal()
        masked_bits = masked.bit_decompose(bit_length,step=step)
        return [m + r for m,r in zip(masked_bits, random_bits)]

    @vectorize
    def bit_decompose_embedding(self):
        random_bits = [self.get_random_bit() \
                           for i in range(8)]
        one = cgf2n(1)
        wanted_positions = [0, 5, 10, 15, 20, 25, 30, 35]
        masked = sum([b * (one << wanted_positions[i]) for i,b in enumerate(random_bits)], self).reveal()
        return [self.clear_type((masked >> wanted_positions[i]) & one) + r for i,r in enumerate(random_bits)]

for t in (sint, sgf2n):
    t.basic_type = t
    t.default_type = t

sint.bit_type = sintbit
sgf2n.bit_type = sgf2n

class _bitint(object):
    bits = None
    log_rounds = False
    linear_rounds = False

    @staticmethod
    def half_adder(a, b):
        return a.half_adder(b)

    @classmethod
    def bit_adder(cls, a, b, carry_in=0, get_carry=False):
        a, b = list(a), list(b)
        a += [0] * (len(b) - len(a))
        b += [0] * (len(a) - len(b))
        return cls.bit_adder_selection(a, b, carry_in=carry_in,
                                       get_carry=get_carry)

    @classmethod
    def bit_adder_selection(cls, a, b, carry_in=0, get_carry=False):
        if cls.log_rounds:
            return cls.carry_lookahead_adder(a, b, carry_in=carry_in,
                                             get_carry=get_carry)
        elif cls.linear_rounds:
            return cls.ripple_carry_adder(a, b, carry_in=carry_in,
                                             get_carry=get_carry)
        else:
            return cls.carry_select_adder(a, b, carry_in=carry_in,
                                          get_carry=get_carry)

    @classmethod
    def carry_lookahead_adder(cls, a, b, fewer_inv=False, carry_in=0,
                              get_carry=False):
        lower = []
        a, b = a[:], b[:]
        for (ai, bi) in zip(a[:], b[:]):
            if is_zero(ai) or is_zero(bi):
                lower.append(ai + bi)
                a.pop(0)
                b.pop(0)
            else:
                break
        carries = cls.get_carries(a, b, fewer_inv=fewer_inv, carry_in=carry_in)
        res = lower + cls.sum_from_carries(a, b, carries)
        if get_carry:
            res += [carries[-1]]
        return res

    @classmethod
    def get_carries(cls, a, b, fewer_inv=False, carry_in=0):
        d = [cls.half_adder(ai, bi) for (ai,bi) in zip(a,b)]
        carry = floatingpoint.carry
        if fewer_inv:
            pre_op = floatingpoint.PreOpL2
        else:
            pre_op = floatingpoint.PreOpL
        if d:
            carries = list(zip(*pre_op(carry, [(0, carry_in)] + d)))[1]
        else:
            carries = []
        return carries

    @staticmethod
    def sum_from_carries(a, b, carries):
        return [ai.bit_xor(bi).bit_xor(carry) \
                for (ai, bi, carry) in zip(a, b, carries)]

    @classmethod
    def carry_select_adder(cls, a, b, get_carry=False, carry_in=0):
        a += [0] * (len(b) - len(a))
        b += [0] * (len(a) - len(b))
        n = len(a)
        for m in range(100):
            if sum(range(m + 1)) + 1 >= n:
                break
        for k in range(m, -1, -1):
            if sum(range(m, k - 1, -1)) + 1 >= n:
                break
        blocks = list(range(m, k, -1))
        blocks.append(n - sum(blocks))
        blocks.reverse()
        #print 'blocks:', blocks
        if len(blocks) > 1 and blocks[0] > blocks[1]:
            raise Exception('block size not increasing:', blocks)
        if sum(blocks) != n:
            raise Exception('blocks not summing up: %s != %s' % \
                            (sum(blocks), n))
        res = []
        carry = carry_in
        cin_one = util.long_one(a + b)
        for m in blocks:
            aa = a[:m]
            bb = b[:m]
            a = a[m:]
            b = b[m:]
            cc = [cls.ripple_carry_adder(aa, bb, i) for i in (0, cin_one)]
            for i in range(m):
                res.append(util.if_else(carry, cc[1][i], cc[0][i]))
            carry = util.if_else(carry, cc[1][m], cc[0][m])
        if get_carry:
            res += [carry]
        return res

    @classmethod
    def ripple_carry_adder(cls, a, b, carry_in=0, get_carry=True):
        carry = carry_in
        res = []
        for aa, bb in zip(a, b):
            cc, carry = cls.full_adder(aa, bb, carry)
            res.append(cc)
        if get_carry:
            res.append(carry)
        return res

    @staticmethod
    def full_adder(a, b, carry):
        s = a ^ b
        return s ^ carry, a ^ (s & (carry ^ a))

    @staticmethod
    def bit_comparator(a, b):
        long_one = util.long_one(a + b)
        op = lambda y,x,*args: (util.if_else(x[1], x[0], y[0]), \
                                    util.if_else(x[1], long_one, y[1]))
        return floatingpoint.KOpL(op, [(bi, ai + bi) for (ai,bi) in zip(a,b)])        

    @classmethod
    def bit_less_than(cls, a, b):
        x, not_equal = cls.bit_comparator(a, b)
        return util.if_else(not_equal, x, 0)

    @staticmethod
    def get_highest_different_bits(a, b, index):
        diff = [ai + bi for (ai,bi) in reversed(list(zip(a,b)))]
        preor = floatingpoint.PreOR(diff, raw=True)
        highest_diff = [x - y for (x,y) in reversed(list(zip(preor, [0] + preor)))]
        raw = sum(map(operator.mul, highest_diff, (a,b)[index]))
        return raw.bit_decompose()[0]

    def add(self, other):
        if type(other) == self.bin_type:
            raise CompilerError('Unclear addition')
        a = self.bit_decompose()
        b = util.bit_decompose(other, self.n_bits)
        return self.compose(self.bit_adder(a, b))

    @ret_cisc
    def mul(self, other):
        if type(other) == self.bin_type:
            raise CompilerError('Unclear multiplication')
        self_bits = self.bit_decompose()
        if isinstance(other, int):
            other_bits = util.bit_decompose(other, self.n_bits)
            bit_matrix = [[x * y for y in self_bits] for x in other_bits]
        else:
            try:
                other_bits = other.bit_decompose()
                if len(other_bits) == 1:
                    return type(self)(other_bits[0] * self)
                if len(self_bits) != len(other_bits):
                   raise NotImplementedError('Multiplication of different lengths')
            except AttributeError:
                pass
            try:
                other = self.bin_type(other)
            except CompilerError:
                return NotImplemented
            bit_matrix = self.get_bit_matrix(self_bits, other)
        return self.compose(self.wallace_tree_from_matrix(bit_matrix, False))

    @classmethod
    def wallace_tree_from_matrix(cls, bit_matrix, get_carry=True):
        columns = [[_f for _f in (bit_matrix[j][i-j] \
                                     for j in range(min(len(bit_matrix), i + 1))) \
                    if not is_zero(_f)] \
                       for i in range(len(bit_matrix[0]))]
        return cls.wallace_tree_from_columns(columns, get_carry)

    @classmethod
    def wallace_tree_without_finish(cls, columns, get_carry=True):
        self = cls
        columns = [col[:] for col in columns]
        while max(len(c) for c in columns) > 2:
            new_columns = [[] for i in range(len(columns) + 1)]
            for i,col in enumerate(columns):
                while len(col) > 2:
                    s, carry = self.full_adder(*(col.pop() for i in range(3)))
                    new_columns[i].append(s)
                    new_columns[i+1].append(carry)
                if len(col) == 2:
                    s, carry = self.half_adder(*(col.pop() for i in range(2)))
                    new_columns[i].append(s)
                    new_columns[i+1].append(carry)
                else:
                    new_columns[i].extend(col)
            if get_carry:
                columns = new_columns
            else:
                columns = new_columns[:-1]
        for col in columns:
            col.extend([0] * (2 - len(col)))
        return tuple(list(x) for x in zip(*columns))

    @classmethod
    def wallace_tree_from_columns(cls, columns, get_carry=True):
        summands = cls.wallace_tree_without_finish(columns, get_carry)
        return cls.bit_adder(*summands)

    @classmethod
    def wallace_tree(cls, rows):
        return cls.wallace_tree_from_columns([list(x) for x in zip(*rows)])

    @classmethod
    def wallace_reduction(cls, a, b, c, get_carry=True):
        assert len(a) == len(b) == len(c)
        tmp = zip(*(cls.full_adder(*x) for x in zip(a, b, c)))
        sums, carries = (list(x) for x in tmp)
        carries = [0] + carries
        if get_carry:
            sums += [0]
        else:
            del carries[-1]
        return sums, carries

    def __sub__(self, other):
        if type(other) == sgf2n:
            raise CompilerError('Unclear subtraction')
        a = self.bit_decompose()
        b = util.bit_decompose(other, self.n_bits)
        from util import bit_not, bit_and, bit_xor
        n = 1
        for x in (a + b):
            try:
                n = x.n
                break
            except:
                pass
        d = [(bit_not(bit_xor(ai, bi), n), bit_and(bit_not(ai, n), bi))
             for (ai,bi) in zip(a,b)]
        borrow = lambda y,x,*args: \
            (bit_and(x[0], y[0]), util.OR(x[1], bit_and(x[0], y[1])))
        borrows = (0,) + list(zip(*floatingpoint.PreOpL(borrow, d)))[1]
        return self.compose(reduce(util.bit_xor, (ai, bi, borrow)) \
                                for (ai,bi,borrow) in zip(a,b,borrows))

    def __rsub__(self, other):
        raise NotImplementedError()

    def __truediv__(self, other):
        raise NotImplementedError()

    def __truerdiv__(self, other):
        raise NotImplementedError()

    def __lshift__(self, other):
        return self.compose(([0] * other + self.bit_decompose())[:self.n_bits])

    def __rshift__(self, other):
        return self.compose(self.bit_decompose()[other:])

    def bit_decompose(self, n_bits=None, security=None):
        if self.bits is None:
            self.bits = self.force_bit_decompose(self.n_bits)
        if n_bits is None:
            return self.bits[:]
        else:
            return self.bits[:n_bits] + [self.fill_bit()] * (n_bits - self.n_bits)

    def fill_bit(self):
        return self.bits[-1]

    @staticmethod
    def prep_comparison(a, b):
        a[-1], b[-1] = b[-1], a[-1]
    
    def comparison(self, other, const_rounds=False, index=None):
        a = self.bit_decompose()
        b = util.bit_decompose(other, self.n_bits)
        self.prep_comparison(a, b)
        if const_rounds:
            return self.get_highest_different_bits(a, b, index)
        else:
            return self.bit_comparator(a, b)

    def __lt__(self, other):
        if program.options.comparison == 'log':
            x, not_equal = self.comparison(other)
            return util.if_else(not_equal, x, 0)
        else:
            return self.comparison(other, True, 1)

    def __le__(self, other):
        if program.options.comparison == 'log':
            x, not_equal = self.comparison(other)
            return util.if_else(not_equal, x, 1)
        else:
            return 1 - self.comparison(other, True, 0)

    def __ge__(self, other):
        return 1 - (self < other)

    def __gt__(self, other):
        return 1 - (self <= other)

    def __eq__(self, other, bit_length=None, security=None):
        diff = self ^ other
        diff_bits = [1 - x for x in diff.bit_decompose()[:bit_length]]
        return floatingpoint.KMul(diff_bits)

    def __ne__(self, other):
        return 1 - (self == other)

    equal = __eq__

    def __neg__(self):
        bits = self.bit_decompose()
        n = 1
        for b in bits:
            try:
                n = x.n
                break
            except:
                pass
        return 1 + self.compose(util.bit_not(b, n) for b in bits)

    def __abs__(self):
        return util.if_else(self.bit_decompose()[-1], -self, self)

    less_than = lambda self, other, *args, **kwargs: self < other
    greater_than = lambda self, other, *args, **kwargs: self > other
    less_equal = lambda self, other, *args, **kwargs: self <= other
    greater_equal = lambda self, other, *args, **kwargs: self >= other
    equal = lambda self, other, *args, **kwargs: self == other
    not_equal = lambda self, other, *args, **kwargs: self != other

class intbitint(_bitint, sint):
    @staticmethod
    def full_adder(a, b, carry):
        s = a.bit_xor(b)
        return s.bit_xor(carry), util.if_else(s, carry, a)

    @staticmethod
    def sum_from_carries(a, b, carries):
        return [a[i] + b[i] + carries[i] - 2 * carries[i + 1] \
                for i in range(len(a))]

    @classmethod
    def bit_adder_selection(cls, a, b, carry_in=0, get_carry=False):
        if cls.linear_rounds:
            return cls.ripple_carry_adder(a, b, carry_in=carry_in)
        # experimental cut-off with dead code elimination
        elif len(a) < 122 or cls.log_rounds:
            return cls.carry_lookahead_adder(a, b, carry_in=carry_in,
                                             get_carry=get_carry)
        else:
            return cls.carry_select_adder(a, b, carry_in=carry_in)

class sgf2nint(_bitint, sgf2n):
    bin_type = sgf2n

    @classmethod
    def compose(cls, bits):
        bits = list(bits)
        if len(bits) > cls.n_bits:
            raise CompilerError('Too many bits')
        res = cls()
        res.bits = bits + [0] * (cls.n_bits - len(bits))
        gmovs(res, sum(b << i for i,b in enumerate(bits)))
        return res

    @staticmethod
    def get_bit_matrix(self_bits, other):
        products = [x * other for x in self_bits]
        return [util.bit_decompose(x, len(self_bits)) for x in products]

    def load_int(self, other):
        if -2**(self.n_bits-1) <= other < 2**(self.n_bits-1):
            self.bin_type.load_int(self, other + 2**self.n_bits \
                                   if other < 0 else other)
        else:
            raise CompilerError('Invalid signed %d-bit integer: %d' % \
                                    (self.n_bits, other))

    def load_other(self, other):
        if isinstance(other, sgf2nint):
            gmovs(self, self.compose(other.bit_decompose(self.n_bits)))
        elif isinstance(other, sgf2n):
            gmovs(self, other)
        else:
            gaddm(self, sgf2n(0), cgf2n(other))

    def force_bit_decompose(self, n_bits=None):
        return sgf2n(self).bit_decompose(n_bits)

class sgf2nuint(sgf2nint):
    def load_int(self, other):
        if 0 <= other < 2**self.n_bits:
            sgf2n.load_int(self, other)
        else:
            raise CompilerError('Invalid unsigned %d-bit integer: %d' % \
                                    (self.n_bits, other))

    def fill_bit(self):
        return 0

    @staticmethod
    def prep_comparison(a, b):
        pass

class sgf2nuint32(sgf2nuint):
    n_bits = 32

class sgf2nint32(sgf2nint):
    n_bits = 32

def get_sgf2nint(n):
    class sgf2nint_spec(sgf2nint):
        n_bits = n
    #sgf2nint_spec.__name__ = 'sgf2unint' + str(n)
    return sgf2nint_spec

def get_sgf2nuint(n):
    class sgf2nuint_spec(sgf2nint):
        n_bits = n
    #sgf2nuint_spec.__name__ = 'sgf2nuint' + str(n)
    return sgf2nuint_spec

class sgf2nfloat(sgf2n):
    @classmethod
    def set_precision(cls, vlen, plen):
        cls.vlen = vlen
        cls.plen = plen
        class v_type(sgf2nuint):
            n_bits = 2 * vlen + 1
        class p_type(sgf2nint):
            n_bits = plen
        class pdiff_type(sgf2nuint):
            n_bits = plen
        cls.v_type = v_type
        cls.p_type = p_type
        cls.pdiff_type = pdiff_type

    def __init__(self, val, p=None, z=None, s=None):
        super(sgf2nfloat, self).__init__()
        if p is None and type(val) == sgf2n:
            bits = val.bit_decompose(self.vlen + self.plen + 1)
            self.v = self.v_type.compose(bits[:self.vlen])
            self.p = self.p_type.compose(bits[self.vlen:-1])
            self.s = bits[-1]
            self.z = util.tree_reduce(operator.mul, (1 - b for b in self.v.bits))
        else:
            if p is None:
                v, p, z, s = sfloat.convert_float(val, self.vlen, self.plen)
                # correct sfloat
                p += self.vlen - 1
                v_bits = util.bit_decompose(v, self.vlen)
                p_bits = util.bit_decompose(p, self.plen)
                self.v = self.v_type.compose(v_bits)
                self.p = self.p_type.compose(p_bits)
                self.z = z
                self.s = s
            else:
                self.v, self.p, self.z, self.s = val, p, z, s
                v_bits = val.bit_decompose()[:self.vlen]
                p_bits = p.bit_decompose()[:self.plen]
            gmovs(self, util.bit_compose(v_bits + p_bits + [self.s]))

    def add(self, other):
        a = self.p < other.p
        b = self.p == other.p
        c = self.v < other.v
        other_dominates = (b.if_else(c, a))
        pmax, pmin = a.cond_swap(self.p, other.p, self.p_type)
        vmax, vmin = other_dominates.cond_swap(self.v, other.v, self.v_type)
        s3 = self.s ^ other.s
        pdiff = self.pdiff_type(pmax - pmin)
        d = self.vlen < pdiff
        pow_delta = util.pow2(d.if_else(0, pdiff).bit_decompose(util.log2(self.vlen)))
        v3 = vmax
        v4 = self.v_type(sgf2n(vmax) * pow_delta) + self.v_type(s3.if_else(-vmin, vmin))
        v = self.v_type(sgf2n(d.if_else(v3, v4) << self.vlen) / pow_delta)
        v >>= self.vlen - 1
        h = floatingpoint.PreOR(v.bits[self.vlen+1::-1])
        tmp = sum(util.if_else(b, 0, 1 << i) for i,b in enumerate(h))
        pow_p0 = 1 + self.v_type(tmp)
        v = (v * pow_p0) >> 2
        p = pmax - sum(self.p_type.compose([1 - b]) for b in h) + 1
        v = self.z.if_else(other.v, other.z.if_else(self.v, v))
        z = v == 0
        p = z.if_else(0, self.z.if_else(other.p, other.z.if_else(self.p, p)))
        s = other_dominates.if_else(other.s, self.s)
        s = self.z.if_else(other.s, other.z.if_else(self.s, s))
        return sgf2nfloat(v, p, z, s)

    def mul(self, other):
        v = (self.v * other.v) >> (self.vlen - 1)
        b = v.bits[self.vlen]
        v = b.if_else(v >> 1, v)
        p = self.p + other.p + self.p_type.compose([b])
        s = self.s + other.s
        z = util.or_op(self.z, other.z)
        return sgf2nfloat(v, p, z, s)

sgf2nfloat.set_precision(24, 8)

def parse_type(other, k=None, f=None):
    # converts type to cfix/sfix depending on the case
    if isinstance(other, cfix.scalars):
        return cfix(other, k=k, f=f)
    elif isinstance(other, cint):
        tmp = cfix(k=k, f=f)
        tmp.load_int(other)
        return tmp
    elif isinstance(other, sint):
        tmp = sfix(k=k, f=f)
        tmp.load_int(other)
        return tmp
    elif isinstance(other, sfloat):
        tmp = sfix(other, k=k, f=f)
        return tmp
    else:
        return other

class cfix(_number, _structure):
    """ Clear fixed-point number represented as clear integer. """
    __slots__ = ['value', 'f', 'k']
    reg_type = 'c'
    scalars = (int, float, regint, cint)
    @classmethod
    def set_precision(cls, f, k = None):
        """ Set the precision of the integer representation. Note that some
        operations are undefined when the precision of :py:class:`sfix` and
        :py:class:`cfix` differs. The initial defaults are chosen to
        allow the best optimization of probabilistic truncation in
        computation modulo 2^64 (2*k < 64). Generally, 2*k must be at
        most the integer length for rings and at most m-s-1 for
        computation modulo an m-bit prime and statistical security s
        (default 40).

        :param f: bit length of decimal part (initial default 16)
        :param k: whole bit length of fixed point, defaults to twice :py:obj:`f` if not given (initial default 31)

        """
        cls.f = f
        if k is None:
            cls.k = 2 * f
        else:
            cls.k = k

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._new(cint.load_mem(address))

    @vectorized_classmethod
    def read_from_socket(cls, client_id, n=1):
        """
        Receive clear fixed-point value(s) from client. The client needs
        to convert the values to the right integer representation.

        :param client_id: Client id (regint)
        :param n: number of values (default 1)
        :returns: cfix (if n=1) or list of cfix
        """
        cint_input = cint.read_from_socket(client_id, n)
        if n == 1:
            return cfix._new(cint_inputs)
        else:
            return list(map(cfix, cint_inputs))
        
    @vectorized_classmethod
    def write_to_socket(self, client_id, values, message_type=ClientMessageType.NoType):
        """ Send a list of clear fixed-point values to a client
        (represented as clear integers).

        :param client_id: Client id (regint)
        :param values: list of cint
        """
        def cfix_to_cint(fix_val):
            return cint(fix_val.v)
        cint_values = list(map(cfix_to_cint, values))
        writesocketc(client_id, message_type, *cint_values)

    @staticmethod
    def malloc(size, creator_tape=None):
        return program.malloc(size, cint, creator_tape=creator_tape)

    @staticmethod
    def n_elements():
        return 1

    @classmethod
    def from_int(cls, other):
        res = cls()
        res.load_int(other)
        return res

    @classmethod
    def _new(cls, other, k=None, f=None):
        res = cls(k=k, f=f)
        res.v = cint.conv(other)
        return res

    @staticmethod
    def int_rep(v, f, k=None):
        if isinstance(v, regint):
            v = cint(v)
        res = v * (2 ** f)
        try:
            res = int(round(res))
            if k and res >= 2 ** (k - 1) or res < -2 ** (k - 1):
                limit = 2 ** (k - f - 1)
                raise CompilerError(
                    'Value out of fixed-point range [-%d, %d). '
                    'Use `sfix.set_precision(f, k)` with k being at least f+%d'
                    % (limit, limit, res.bit_length() - f + 1))
        except TypeError:
            pass
        return res

    @vectorize_init
    @read_mem_value
    def __init__(self, v=None, k=None, f=None, size=None):
        """ :param v: cfix/float/int """
        f = self.f if f is None else f
        k = self.k if k is None else k
        self.f = f
        self.k = k
        if isinstance(v, cfix.scalars):
            v = self.int_rep(v, f=f, k=k)
            self.v = cint(v, size=size)
        elif isinstance(v, cfix):
            self.v = v.v
        elif v is None:
            self.v = cint(0)
        else:
            raise CompilerError('cannot initialize cfix with %s' % v)

    def __iter__(self):
        for x in self.v:
            yield self._new(x, self.k, self.f)

    def __len__(self):
        return len(self.v)

    @vectorize
    def load_int(self, v):
        self.v = cint(v) * (2 ** self.f)

    @classmethod
    def conv(cls, other):
        if isinstance(other, cls):
            return other
        else:
            try:
                res = cfix()
                res.load_int(other)
                return res
            except (TypeError, CompilerError):
                pass
        return cls(other)

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self.v.store_in_mem(address)

    @property
    def size(self):
        return self.v.size

    def sizeof(self):
        return self.size * 4

    @vectorize
    def add(self, other):
        """ Clear fixed-point addition.

        :param other: cfix/cint/regint/int """
        other = parse_type(other)
        if isinstance(other, cfix):
            return cfix._new(self.v + other.v)
        else:
            return NotImplemented

    @vectorize 
    def mul(self, other):
        """ Clear fixed-point multiplication.

        :param other: cfix/cint/regint/int/sint """
        if isinstance(other, sint):
            return sfix._new(self.v * other, k=self.k, f=self.f)
        if isinstance(other, (int, regint, cint)):
            return cfix._new(self.v * cint(other), k=self.k, f=self.f)
        other = parse_type(other)
        if isinstance(other, cfix):
            assert self.f == other.f
            sgn = cint(1 - 2 * ((self < 0) ^ (other < 0)))
            absolute = self.v * other.v * sgn
            val = sgn * (absolute >> self.f)
            return cfix._new(val)
        elif isinstance(other, sfix):
            return NotImplemented
        else:
            raise CompilerError('Invalid type %s for cfix.__mul__' % type(other))

    def positive_mul(self, other):
        assert isinstance(other, float)
        assert other >= 0
        v = self.v * int(round(other * 2 ** self.f))
        return self._new(v >> self.f, k=self.k, f=self.f)

    @vectorize
    def __sub__(self, other):
        """ Clear fixed-point subtraction.

        :param other: cfix/cint/regint/int """
        other = parse_type(other)
        if isinstance(other, cfix):
            return cfix._new(self.v - other.v)
        elif isinstance(other, sfix):
            return sfix._new(self.v - other.v)
        else:
            raise NotImplementedError

    @vectorize
    def __neg__(self):
        """ Clear fixed-point negation. """
        # cfix type always has .v
        return cfix._new(-self.v)
    
    def __rsub__(self, other):
        return -self + other
    __rsub__.__doc__ = __sub__.__doc__

    @vectorize
    def __eq__(self, other):
        """ Clear fixed-point comparison.

        :param other: cfix/cint/regint/int
        :return: 0/1
        :rtype: regint """
        other = parse_type(other)
        if isinstance(other, cfix):
            return self.v == other.v
        elif isinstance(other, sfix):
            return other.v.equal(self.v, self.k, other.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __lt__(self, other):
        """ Clear fixed-point comparison. """
        other = parse_type(other)
        if isinstance(other, cfix):
            assert self.k == other.k
            return self.v.less_than(other.v, self.k)
        elif isinstance(other, sfix):
            if(self.k != other.k or self.f != other.f):
                raise TypeError('Incompatible fixed point types in comparison')
            return other.v.greater_than(self.v, self.k, other.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __le__(self, other):
        """ Clear fixed-point comparison. """
        other = parse_type(other)
        if isinstance(other, cfix):
            return 1 - (self > other)
        elif isinstance(other, sfix):
            return other.v.greater_equal(self.v, self.k, other.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __gt__(self, other):
        """ Clear fixed-point comparison. """
        other = parse_type(other)
        if isinstance(other, cfix):
            return other.__lt__(self)
        elif isinstance(other, sfix):
            return other.v.less_than(self.v, self.k, other.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __ge__(self, other):
        """ Clear fixed-point comparison. """
        other = parse_type(other)
        if isinstance(other, cfix):
            return 1 - (self < other)
        elif isinstance(other, sfix):
            return other.v.less_equal(self.v, self.k, other.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __ne__(self, other):
        """ Clear fixed-point comparison. """
        other = parse_type(other)
        if isinstance(other, cfix):
            return self.v != other.v
        elif isinstance(other, sfix):
            return other.v.not_equal(self.v, self.k, other.kappa)
        else:
            raise NotImplementedError

    for op in __le__, __lt__, __ge__, __gt__, __ne__:
        op.__doc__ = __eq__.__doc__
    del op

    @vectorize
    def __truediv__(self, other):
        """ Clear fixed-point division.

        :param other: cfix/cint/regint/int """
        other = parse_type(other, self.k, self.f)
        if isinstance(other, cfix):
            return cfix._new(library.cint_cint_division(
                self.v, other.v, self.k, self.f), k=self.k, f=self.f)
        elif isinstance(other, sfix):
            assert self.k == other.k
            assert self.f == other.f
            return sfix._new(library.FPDiv(self.v, other.v, self.k, self.f,
                                           other.kappa,
                                           nearest=sfix.round_nearest),
                             k=self.k, f=self.f)
        else:
            raise TypeError('Incompatible fixed point types in division')

    @vectorize
    def __rtruediv__(self, other):
        """ Fixed-point division.

        :param other: sfix/sint/cfix/cint/regint/int """
        other = parse_type(other, self.k, self.f)
        return other / self

    def print_plain(self):
        """ Clear fixed-point output. """
        print_float_plain(cint.conv(self.v), cint(-self.f), \
                          cint(0), cint(0), cint(0))

    def output_if(self, cond):
        cond_print_plain(cond, self.v, cint(-self.f))

class _single(_number, _structure):
    """ Representation as single integer preserving the order """
    """ E.g. fixed-point numbers """
    __slots__ = ['v']
    kappa = None
    round_nearest = False
    """ Whether to round deterministically to nearest instead of
    probabilistically, e.g. after fixed-point multiplication. """

    @classmethod
    def receive_from_client(cls, n, client_id, message_type=ClientMessageType.NoType):
        """
        Securely obtain shares of values input by a client. Assumes client
        has already converted values to integer representation.

        :param n: number of inputs (int)
        :param client_id: regint

        """
        sint_inputs = cls.int_type.receive_from_client(n, client_id, ClientMessageType.TripleShares)
        return list(map(cls, sint_inputs))

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        return cls._new(cls.int_type.load_mem(address))

    @classmethod
    @read_mem_value
    def conv(cls, other):
        if isinstance(other, cls):
            return other
        else:
            try:
                return cls.from_sint(other)
            except (TypeError, CompilerError):
                pass
        return cls(other)

    @classmethod
    def coerce(cls, other):
        return cls.conv(other)

    @classmethod
    def malloc(cls, size, creator_tape=None):
        return cls.int_type.malloc(size, creator_tape=creator_tape)

    @classmethod
    def free(cls, addr):
        return cls.int_type.free(addr)

    @classmethod
    def n_elements(cls):
        return cls.int_type.n_elements()

    @classmethod
    def mem_size(cls):
        return cls.int_type.mem_size()

    @classmethod
    def dot_product(cls, x, y, res_params=None):
        """ Secret dot product.

        :param x: iterable of appropriate secret type
        :param y: iterable of appropriate secret type and same length """
        return cls.unreduced_dot_product(x, y, res_params).reduce_after_mul()

    @classmethod
    def unreduced_dot_product(cls, x, y, res_params=None):
        dp = cls.int_type.dot_product([xx.pre_mul() for xx in x],
                                      [yy.pre_mul() for yy in y])
        return x[0].unreduced(dp, y[0], res_params, len(x))

    @classmethod
    def row_matrix_mul(cls, row, matrix, res_params=None):
        int_matrix = [y.get_vector().pre_mul() for y in matrix]
        col = cls.int_type.row_matrix_mul([x.pre_mul() for x in row],
                                          int_matrix)
        res = row[0].unreduced(col, matrix[0][0], res_params,
                               len(row)).reduce_after_mul()
        return res

    @classmethod
    def matrix_mul(cls, A, B, n, res_params=None):
        AA = A.pre_mul()
        BB = B.pre_mul()
        CC = cls.int_type.matrix_mul(AA, BB, n)
        res = A.unreduced(CC, B, res_params, n).reduce_after_mul()
        return res

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        self.v.store_in_mem(address)

    @property
    def size(self):
        return self.v.size

    def sizeof(self):
        return self.size

    def __len__(self):
        """ Vector length. """
        return len(self.v)

    @vectorize
    def __sub__(self, other):
        """ Subtraction.

        :param other: appropriate public or secret (incl. sint/cint/regint/int) """
        other = self.coerce(other)
        return self + (-other)

    def __rsub__(self, other):
        return -self + other
    __rsub__.__doc__ = __sub__.__doc__

    @vectorize
    def __eq__(self, other):
        """ Comparison.

        :param other: appropriate public or secret (incl. sint/cint/regint/int)
        :return: 0/1
        :rtype: same as internal representation"""
        other = self.coerce(other)
        if isinstance(other, (cfix, _single)):
            return self.v.equal(other.v, self.k, self.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __le__(self, other):
        other = self.coerce(other)
        if isinstance(other, (cfix, _single)):
            return self.v.less_equal(other.v, self.k, self.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __lt__(self, other):
        other = self.coerce(other)
        if isinstance(other, (cfix, _single)):
            return self.v.less_than(other.v, self.k, self.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __ge__(self, other):
        other = self.coerce(other)
        if isinstance(other, (cfix, _single)):
            return self.v.greater_equal(other.v, self.k, self.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __gt__(self, other):
        other = self.coerce(other)
        if isinstance(other, (cfix, _single)):
            return self.v.greater_than(other.v, self.k, self.kappa)
        else:
            raise NotImplementedError

    @vectorize
    def __ne__(self, other):
        other = self.coerce(other)
        if isinstance(other, (cfix, _single)):
            return self.v.not_equal(other.v, self.k, self.kappa)
        else:
            raise NotImplementedError

    for op in __le__, __lt__, __ge__, __gt__, __ne__:
        op.__doc__ = __eq__.__doc__
    del op

    def link(self, other):
        self.v.link(other.v)

class _fix(_single):
    """ Secret fixed point type. """
    __slots__ = ['v', 'f', 'k']

    def set_precision(cls, f, k = None):
        cls.f = f
        # default bitlength = 2*precision
        if k is None:
            cls.k = 2 * f
        else:
            cls.k = k
    set_precision.__doc__ = cfix.set_precision.__doc__
    set_precision = classmethod(set_precision)

    @classmethod
    def set_precision_from_args(cls, program, adapt_ring=False):
        f = None
        k = None
        for arg in program.args:
            m = re.match('f([0-9]+)$', arg)
            if m:
                f = int(m.group(1))
            m = re.match('k([0-9]+)$', arg)
            if m:
                k = int(m.group(1))
        if f is not None:
            print ('Setting fixed-point precision to %d/%s' % (f, k))
            cls.set_precision(f, k)
            cfix.set_precision(f, k)
        elif k is not None:
            raise CompilerError('need to set fractional precision')
        if 'nearest' in program.args:
            print('Nearest rounding instead of proabilistic '
                  'for fixed-point computation')
            cls.round_nearest = True
        if adapt_ring and program.options.ring:
            need = 2 ** int(math.ceil(math.log(2 * cls.k, 2)))
            if need != int(program.options.ring):
                print('Changing computation modulus to 2^%d' % need)
                program.set_ring_size(need)

    @classmethod
    def coerce(cls, other):
        if isinstance(other, (_fix, cls.clear_type)):
            return other
        else:
            return cls.conv(other)

    @classmethod
    def from_sint(cls, other, k=None, f=None):
        """ Convert secret integer.

        :param other: sint """
        res = cls(k=k, f=f)
        res.load_int(cls.int_type.conv(other))
        return res

    @classmethod
    def _new(cls, other, k=None, f=None):
        res = cls(k=k, f=f)
        res.v = cls.int_type.conv(other)
        return res

    @vectorize_init
    def __init__(self, _v=None, k=None, f=None, size=None):
        """ :params _v: int/float/regint/cint/sint/sfloat """
        if k is None:
            k = self.k
        else:
            self.k = k
        if f is None:
            f = self.f
        else:
            self.f = f
        assert k is not None
        assert f is not None
        if _v is None:
            self.v = self.int_type(0)
        elif isinstance(_v, self.int_type):
            self.load_int(_v)
        elif isinstance(_v, cfix.scalars):
            self.v = self.int_type(cfix.int_rep(_v, f=f, k=k), size=size)
        elif isinstance(_v, self.float_type):
            p = (f + _v.p)
            b = (p.greater_equal(0, _v.vlen))
            a = b*(_v.v << (p)) + (1-b)*(_v.v >> (-p))
            self.v = (1-2*_v.s)*a
        elif isinstance(_v, type(self)):
            self.v = _v.v
        elif isinstance(_v, (MemValue, MemFix)):
            #this is a memvalue object
            self.v = type(self)(_v.read()).v
        else:
            raise CompilerError('cannot convert %s to sfix' % _v)
        if not isinstance(self.v, self.int_type):
            raise CompilerError('sfix conversion failure: %s/%s' % (_v, self.v))

    def load_int(self, v):
        self.v = self.int_type(v) << self.f

    def __getitem__(self, index):
        return self._new(self.v[index])

    @vectorize 
    def add(self, other):
        """ Secret fixed-point addition.

        :param other: sfix/cfix/sint/cint/regint/int """
        other = self.coerce(other)
        if isinstance(other, (_fix, cfix)):
            return self._new(self.v + other.v, k=self.k, f=self.f)
        elif isinstance(other, cfix.scalars):
            tmp = cfix(other, k=self.k, f=self.f)
            return self + tmp
        else:
            return NotImplemented

    def mul(self, other):
        """ Secret fixed-point multiplication.

        :param other: sfix/cfix/sint/cint/regint/int """
        if isinstance(other, (sint, cint, regint, int)):
            return self._new(self.v * other, k=self.k, f=self.f)
        elif isinstance(other, float):
            if int(other) == other:
                return self.mul(int(other))
            v = int(round(other * 2 ** self.f))
            if v == 0:
                return 0
            f = self.f
            while v % 2 == 0:
                f -= 1
                v //= 2
            k = len(bin(abs(v))) - 1
            other = self.multipliable(v, k, f, self.size)
        try:
            other = self.coerce(other)
        except:
            return NotImplemented
        if isinstance(other, (_fix, self.clear_type)):
            val = self.v.TruncMul(other.v, self.k + other.k, other.f,
                                  self.kappa,
                                  self.round_nearest)
            if 'vec' not in self.__dict__:
                return self._new(val, k=self.k, f=self.f)
            else:
                return self.vec._new(val, k=self.k, f=self.f)
        elif isinstance(other, cfix.scalars):
            scalar_fix = cfix(other)
            return self * scalar_fix
        else:
            return NotImplemented

    @vectorize
    def __neg__(self):
        """ Secret fixed-point negation. """
        return self._new(-self.v, k=self.k, f=self.f)

    @vectorize
    def __truediv__(self, other):
        """ Secret fixed-point division.

        :param other: sfix/cfix/sint/cint/regint/int """
        if util.is_constant_float(other):
            assert other != 0
            log = math.ceil(math.log(abs(other), 2))
            other_length = self.f + log
            if other_length >= self.k - 1:
                factor = 2 ** (self.k - other_length - 2)
                self *= factor
                other *= factor
            if 2 ** log == other:
                return self * 2 ** -log
        other = self.coerce(other)
        assert self.k == other.k
        assert self.f == other.f
        if isinstance(other, _fix):
            v = library.FPDiv(self.v, other.v, self.k, self.f, self.kappa,
                              nearest=self.round_nearest)
        elif isinstance(other, cfix):
            v = library.sint_cint_division(self.v, other.v, self.k, self.f,
                                           self.kappa)
        else:
            raise TypeError('Incompatible fixed point types in division')
        return self._new(v, k=self.k, f=self.f)

    @vectorize
    def __rtruediv__(self, other):
        """ Secret fixed-point division.

        :param other: sfix/cfix/sint/cint/regint/int """
        return self.coerce(other) / self

    @vectorize
    def compute_reciprocal(self):
        """ Secret fixed-point reciprocal. """
        return type(self)(library.FPDiv(cint(2) ** self.f, self.v, self.k, self.f, self.kappa, True))

    def reveal(self):
        """ Reveal secret fixed-point number.

        :return: relevant clear type """
        val = self.v.reveal()
        class revealed_fix(self.clear_type):
            f = self.f
            k = self.k
        return revealed_fix._new(val)

class sfix(_fix):
    """ Secret fixed-point number represented as secret integer.
    This uses integer operations internally, see :py:class:`sint` for security
    considerations. """
    int_type = sint
    clear_type = cfix

    @vectorized_classmethod
    def get_input_from(cls, player):
        """ Secret fixed-point input.

        :param player: public (regint/cint/int)
        :param size: vector size (int, default 1)
        """
        cls.int_type.require_bit_length(cls.k)
        v = cls.int_type()
        inputmixed('fix', v, cls.f, player)
        return cls._new(v)

    @vectorized_classmethod
    def get_raw_input_from(cls, player):
        return cls._new(cls.int_type.get_raw_input_from(player))

    @vectorized_classmethod
    def get_random(cls, lower, upper):
        """ Uniform secret random number around centre of bounds.
        Actual range can be smaller but never larger.

        :param lower: float
        :param upper: float
        :param size: vector size (int, default 1)
        """
        log_range = int(math.log(upper - lower, 2))
        n_bits = log_range + cls.f
        average = lower + 0.5 * (upper - lower)
        lower = average - 0.5 * 2 ** log_range
        return cls._new(cls.int_type.get_random_int(n_bits)) + lower

    @classmethod
    def direct_matrix_mul(cls, A, B, n, m, l, reduce=True, indices=None):
        # pre-multiplication must be identity
        tmp = cls.int_type.direct_matrix_mul(A, B, n, m, l, indices=indices)
        res = unreduced_sfix._new(tmp)
        if reduce:
            res = res.reduce_after_mul()
        return res

    @classmethod
    def dot_product(cls, x, y, res_params=None):
        """ Secret dot product.

        :param x: iterable of appropriate secret type
        :param y: iterable of appropriate secret type and same length """
        x, y = list(x), list(y)
        if res_params is None:
            if isinstance(x[0], cls.int_type):
                x, y = y, x
            if isinstance(y[0], cls.int_type):
                return cls._new(cls.int_type.dot_product((xx.v for xx in x), y),
                                k=x[0].k, f=x[0].f)
        return super().dot_product(x, y, res_params)

    def expand_to_vector(self, size):
        return self._new(self.v.expand_to_vector(size), k=self.k, f=self.f)

    def coerce(self, other):
        return parse_type(other, k=self.k, f=self.f)

    def mul_no_reduce(self, other, res_params=None):
        assert self.f == other.f
        assert self.k == other.k
        return self.unreduced(self.v * other.v)

    def pre_mul(self):
        return self.v

    def unreduced(self, v, other=None, res_params=None, n_summands=1):
        return unreduced_sfix(v, self.k * 2, self.f, self.kappa)

    @staticmethod
    def multipliable(v, k, f, size):
        return cfix._new(cint.conv(v, size=size), k, f)

    def reveal_to(self, player):
        """ Reveal secret value to :py:obj:`player`.
        Raw representation possibly written to
        ``Player-Data/Private-Output-P<player>``, but not if
        :py:obj:`player` is a :py:class:`regint`.

        :param player: public integer (int/regint/cint)
        :returns: value to be used with :py:func:`~Compiler.library.print_ln_to`
        """
        return personal(player, cfix._new(self.v.reveal_to(player)._v,
                                          self.k, self.f))

class unreduced_sfix(_single):
    int_type = sint

    @classmethod
    def _new(cls, v):
        return cls(v, 2 * sfix.k, sfix.f, sfix.kappa)

    def __init__(self, v, k, m, kappa):
        self.v = v
        self.k = k
        self.m = m
        self.kappa = kappa
        assert self.k is not None
        assert self.m is not None

    def __add__(self, other):
        if is_zero(other):
            return self
        assert self.k == other.k
        assert self.m == other.m
        assert self.kappa == other.kappa
        return unreduced_sfix(self.v + other.v, self.k, self.m, self.kappa)

    __radd__ = __add__

    @vectorize
    def reduce_after_mul(self):
        v = sfix.int_type.round(self.v, self.k, self.m, self.kappa,
                                nearest=sfix.round_nearest, signed=True)
        return sfix._new(v, k=self.k // 2, f=self.m)

sfix.unreduced_type = unreduced_sfix

sfix.set_precision(16, 31)
cfix.set_precision(16, 31)

class squant(_single):
    """ Quantization as in ArXiv:1712.05877v1 """
    __slots__ = ['params']
    int_type = sint
    clamp = True

    @classmethod
    def set_params(cls, S, Z=0, k=8):
        cls.params = squant_params(S, Z, k)

    @classmethod
    def from_sint(cls, other):
        raise CompilerError('sint to squant conversion not implemented')

    @classmethod
    def _new(cls, value, params=None):
        res = cls(params=params)
        res.v = value
        return res

    @read_mem_value
    def __init__(self, value=None, params=None):
        if params is not None:
            self.params = params
        if value is None:
            # need to set v manually
            pass
        elif isinstance(value, cfix.scalars):
            set_global_vector_size(1)
            q = util.round_to_int(value / self.S + self.Z)
            if util.is_constant(q) and (q < 0 or q >= 2**self.k):
                raise CompilerError('%f not quantizable' % value)
            self.v = self.int_type(q)
            reset_global_vector_size()
        elif isinstance(value, squant) and value.params == self.params:
            self.v = value.v
        else:
            raise CompilerError('cannot convert %s to squant' % value)

    def __getitem__(self, index):
        return type(self)._new(self.v[index], self.params)

    def get_params(self):
        return self.params

    @property
    def S(self):
        return self.params.S
    @property
    def Z(self):
        return self.params.Z
    @property
    def k(self):
        return self.params.k

    def coerce(self, other):
        other = self.conv(other)
        return self._new(util.expand(other.v, self.size), other.params)

    @vectorize
    def add(self, other):
        other = self.coerce(other)
        assert self.get_params() == other.get_params()
        return self._new(self.v + other.v - util.expand(self.Z, self.v.size))

    def mul(self, other, res_params=None):
        return self.mul_no_reduce(other, res_params).reduce_after_mul()

    def mul_no_reduce(self, other, res_params=None):
        if isinstance(other, (sint, cint, regint)):
            return self._new(other * (self.v - self.Z) + self.Z,
                             params=self.get_params())
        other = self.coerce(other)
        tmp = (self.v - self.Z) * (other.v - other.Z)
        return _unreduced_squant(tmp, (self.get_params(), other.get_params()),
                                       res_params=res_params)

    def pre_mul(self):
        return self.v - util.expand(self.Z, self.v.size)

    def unreduced(self, v, other, res_params=None, n_summands=1):
        return _unreduced_squant(v, (self.get_params(), other.get_params()),
                                 res_params, n_summands)

    @vectorize
    def for_mux(self, other):
        other = self.coerce(other)
        assert self.params == other.params
        f = lambda x: self._new(x, self.params)
        return f, self.v, other.v

    @vectorize
    def __neg__(self):
        return self._new(-self.v + 2 * util.expand(self.Z, self.v.size))

class _unreduced_squant(object):
    def __init__(self, v, params, res_params=None, n_summands=1):
        self.v = v
        self.params = params
        self.n_summands = n_summands
        self.res_params = res_params or params[0]

    def __add__(self, other):
        if is_zero(other):
            return self
        assert self.params == other.params
        assert self.res_params == other.res_params
        return _unreduced_squant(self.v + other.v, self.params, self.res_params,
                                 self.n_summands + other.n_summands)

    __radd__ = __add__

    def reduce_after_mul(self):
        return squant_params.conv(self.res_params).reduce(self)

class squant_params(object):
    max_n_summands = 2048

    @staticmethod
    def conv(other):
        if isinstance(other, squant_params):
            return other
        else:
            return squant_params(*other)

    def __init__(self, S, Z=0, k=8):
        try:
            self.S = float(S)
        except:
            self.S = S
        self.Z = MemValue.if_necessary(Z)
        self.k = k
        self._store = {}
        if program.options.ring:
            # cheaper probabilistic truncation
            self.max_length = int(program.options.ring) - 1
        else:
            # safe choice for secret shift
            self.max_length = 71

    def __iter__(self):
        yield self.S
        yield self.Z
        yield self.k

    def is_constant(self):
        return util.is_constant_float(self.S) and util.is_constant(self.Z)

    def get(self, input_params, n_summands):
        p = input_params
        M = p[0].S * p[1].S / self.S
        logM = util.log2(M)
        n_shift = self.max_length - p[0].k - p[1].k - util.log2(n_summands)
        if util.is_constant_float(M):
            n_shift -= logM
            int_mult = int(round(M * 2 ** (n_shift)))
        else:
            int_mult = MemValue(M.v << (n_shift + M.p))
        shifted_Z = MemValue.if_necessary(self.Z << n_shift)
        return n_shift, int_mult, shifted_Z

    def precompute(self, *input_params):
        self._store[input_params] = self.get(input_params, self.max_n_summands)

    def get_stored(self, unreduced):
        assert unreduced.n_summands <= self.max_n_summands
        return self._store[unreduced.params]

    def reduce(self, unreduced):
        ps = (self,) + unreduced.params
        if reduce(operator.and_, (p.is_constant() for p in ps)):
            n_shift, int_mult, shifted_Z = self.get(unreduced.params,
                                                    unreduced.n_summands)
        else:
            n_shift, int_mult, shifted_Z = self.get_stored(unreduced)
        size = unreduced.v.size
        n_shift = util.expand(n_shift, size)
        shifted_Z = util.expand(shifted_Z, size)
        int_mult = util.expand(int_mult, size)
        tmp = unreduced.v * int_mult + shifted_Z
        shifted = tmp.round(self.max_length, n_shift,
                            kappa=squant.kappa, nearest=squant.round_nearest,
                            signed=True)
        if squant.clamp:
            length = max(self.k, self.max_length - n_shift) + 1
            top = (1 << self.k) - 1
            over = shifted.greater_than(top, length, squant.kappa)
            under = shifted.less_than(0, length, squant.kappa)
            shifted = over.if_else(top, shifted)
            shifted = under.if_else(0, shifted)
        return squant._new(shifted, params=self)

class sfloat(_number, _structure):
    """
    Secret floating-point number.
    Represents :math:`(1 - 2s) \cdot (1 - z)\cdot v \cdot 2^p`.
        
        v: significand

        p: exponent

        z: zero flag

        s: sign bit

    This uses integer operations internally, see :py:class:`sint` for security
    considerations. """
    __slots__ = ['v', 'p', 'z', 's', 'size']

    # single precision
    vlen = 24
    plen = 8
    kappa = None
    round_nearest = False

    @staticmethod
    def n_elements():
        return 4

    @classmethod
    def malloc(cls, size, creator_tape=None):
        return program.malloc(size * cls.n_elements(), sint,
                              creator_tape=creator_tape)

    @classmethod
    def is_address_tuple(cls, address):
        if isinstance(address, (list, tuple)):
            assert(len(address) == cls.n_elements())
            return True
        return False

    @vectorized_classmethod
    def load_mem(cls, address, mem_type=None):
        """ Load from memory by public address. """
        size = get_global_vector_size()
        if cls.is_address_tuple(address):
            return sfloat(*(sint.load_mem(a, size=size) for a in address))
        res = []
        for i in range(4):
            res.append(sint.load_mem(address + i * size, size=size))
        return sfloat(*res)

    @classmethod
    def set_error(cls, error):
        # incompatible with loops
        #cls.error += error - cls.error * error
        cls.error = error
        pass

    @classmethod
    def conv(cls, other):
        if isinstance(other, cls):
            return other
        else:
            return cls(other)

    @classmethod
    def coerce(cls, other):
        return cls.conv(other)

    @staticmethod
    def convert_float(v, vlen, plen):
        if v < 0:
            s = 1
        else:
            s = 0
        if v == 0:
            v = 0
            p = 0
            z = 1
        else:
            p = int(math.floor(math.log(abs(v), 2))) - vlen + 1
            vv = v
            v = int(round(abs(v) * 2 ** (-p)))
            if v == 2 ** vlen:
                p += 1
                v //= 2
            z = 0
            if p < -2 ** (plen - 1):
                print('Warning: %e truncated to zero' % vv)
                v, p, z = 0, 0, 1
            if p >= 2 ** (plen - 1):
                raise CompilerError('Cannot convert %s to float ' \
                                        'with %d exponent bits' % (vv, plen))
        return v, p, z, s

    @vectorized_classmethod
    def get_input_from(cls, player):
        """ Secret floating-point input.

        :param player: public (regint/cint/int)
        :param size: vector size (int, default 1)
        """
        v = sint()
        p = sint()
        z = sint()
        s = sint()
        inputmixed('float', v, p, z, s, cls.vlen, player)
        return cls(v, p, z, s)

    @vectorize_init
    @read_mem_value
    def __init__(self, v, p=None, z=None, s=None, size=None):
        """
        :param v: initialization (sfloat/sfix/float/int/sint/cint/regint)
        """
        self.size = get_global_vector_size()
        if p is None:
            if isinstance(v, sfloat):
                p = v.p
                z = v.z
                s = v.s
                v = v.v
            elif isinstance(v, sfix):
                f = v.f
                v, p, z, s = floatingpoint.Int2FL(v.v, v.k,
                                                  self.vlen, self.kappa)
                p = p - f
            elif util.is_constant_float(v):
                v, p, z, s = self.convert_float(v, self.vlen, self.plen)
            else:
                v, p, z, s = floatingpoint.Int2FL(sint.conv(v),
                                                  program.bit_length,
                                                  self.vlen, self.kappa)
        if isinstance(v, int):
            if not ((v >= 2**(self.vlen-1) and v < 2**(self.vlen)) or v == 0):
                raise CompilerError('Floating point number malformed: significand')
            self.v = sint(v)
        else:
            self.v = v
        if isinstance(p, int):
            if not (p >= -2**(self.plen - 1) and p < 2**(self.plen - 1)):
                raise CompilerError('Floating point number malformed: exponent %d not unsigned %d-bit integer' % (p, self.plen))
            self.p = sint(p)
        else:
            self.p = p
        if isinstance(z, int):
            if not (z == 0 or z == 1):
                raise CompilerError('Floating point number malformed: zero bit')
            self.z = sint()
            ldsi(self.z, z)
        else:
            self.z = z
        if isinstance(s, int):
            if not (s == 0 or s == 1):
                raise CompilerError('Floating point number malformed: sign')
            self.s = sint()
            ldsi(self.s, s)
        else:
            self.s = s

    def __getitem__(self, index):
        return sfloat(*(x[index] for x in self))

    def __iter__(self):
        yield self.v
        yield self.p
        yield self.z
        yield self.s

    def store_in_mem(self, address):
        """ Store in memory by public address. """
        if self.is_address_tuple(address):
            for a, x in zip(address, self):
                x.store_in_mem(a)
            return
        for i,x in enumerate((self.v, self.p, self.z, self.s)):
            x.store_in_mem(address + i * self.size)

    def sizeof(self):
        return self.size * self.n_elements()

    @vectorize
    def add(self, other):
        """ Secret floating-point addition.

        :param other: sfloat/float/sfix/sint/cint/regint/int """
        other = self.conv(other)
        if isinstance(other, sfloat):
            a,c,d,e = [sint() for i in range(4)]
            t = sint()
            t2 = sint()
            v1 = self.v
            v2 = other.v
            p1 = self.p
            p2 = other.p
            s1 = self.s
            s2 = other.s
            z1 = self.z
            z2 = other.z
            a = p1.less_than(p2, self.plen, self.kappa)
            b = floatingpoint.EQZ(p1 - p2, self.plen, self.kappa)
            c = v1.less_than(v2, self.vlen, self.kappa)
            ap1 = a*p1
            ap2 = a*p2
            aneg = 1 - a
            bneg = 1 - b
            cneg = 1 - c
            av1 = a*v1
            av2 = a*v2
            cv1 = c*v1
            cv2 = c*v2
            pmax = ap2 + p1 - ap1
            pmin = p2 - ap2 + ap1
            vmax = bneg*(av2 + v1 - av1) + b*(cv2 + v1 - cv1)
            vmin = bneg*(av1 + v2 - av2) + b*(cv1 + v2 - cv2)
            s3 = s1 + s2 - 2 * s1 * s2
            comparison.LTZ(d, self.vlen + pmin - pmax + sfloat.round_nearest,
                           self.plen, self.kappa)
            pow_delta = floatingpoint.Pow2((1 - d) * (pmax - pmin),
                                           self.vlen + 1 + sfloat.round_nearest,
                                           self.kappa)
            # deviate from paper for more precision
            #v3 = 2 * (vmax - s3) + 1
            v3 = vmax
            v4 = vmax * pow_delta + (1 - 2 * s3) * vmin
            to_trunc = (d * v3 + (1 - d) * v4)
            if program.options.ring:
                to_trunc <<= 1 + sfloat.round_nearest
                v = floatingpoint.TruncInRing(to_trunc,
                                              2 * (self.vlen + 1 +
                                                   sfloat.round_nearest),
                                              pow_delta)
            else:
                to_trunc *= two_power(self.vlen + sfloat.round_nearest)
                v = to_trunc * floatingpoint.Inv(pow_delta)
                comparison.Trunc(t, v, 2 * self.vlen + 1 + sfloat.round_nearest,
                                 self.vlen - 1, self.kappa, False)
                v = t
            u = floatingpoint.BitDec(v, self.vlen + 2 + sfloat.round_nearest,
                                     self.vlen + 2 + sfloat.round_nearest, self.kappa,
                                     list(range(1 + sfloat.round_nearest,
                                           self.vlen + 2 + sfloat.round_nearest)))
            # using u[0] doesn't seem necessary
            h = floatingpoint.PreOR(u[:sfloat.round_nearest:-1], self.kappa)
            p0 = self.vlen + 1 - sum(h)
            pow_p0 = 1 + sum([two_power(i) * (1 - h[i]) for i in range(len(h))])
            if self.round_nearest:
                t2, overflow = \
                    floatingpoint.TruncRoundNearestAdjustOverflow(pow_p0 * v,
                                                                  self.vlen + 3,
                                                                  self.vlen,
                                                                  self.kappa)
                p0 = p0 - overflow
            else:
                comparison.Trunc(t2, pow_p0 * v, self.vlen + 2, 2, self.kappa, False)
            v = t2
            # deviate for more precision
            #p = pmax - p0 + 1 - d
            p = pmax - p0 + 1
            zz = self.z*other.z
            zprod = 1 - self.z - other.z + zz
            v = zprod*t2 + self.z*v2 + other.z*v1
            z = floatingpoint.EQZ(v, self.vlen, self.kappa)
            p = (zprod*p + self.z*p2 + other.z*p1)*(1 - z)
            s = (1 - b)*(a*other.s + aneg*self.s) + b*(c*other.s + cneg*self.s)
            s = zprod*s + (other.z - zz)*self.s + (self.z - zz)*other.s
            return sfloat(v, p, z, s)
        else:
            return NotImplemented
    
    @vectorize_max
    def mul(self, other):
        """ Secret floating-point multiplication.

        :param other: sfloat/float/sfix/sint/cint/regint/int """
        other = self.conv(other)
        if isinstance(other, sfloat):
            v1 = sint()
            v2 = sint()
            b = sint()
            c2expl = cint()
            comparison.ld2i(c2expl, self.vlen)
            if sfloat.round_nearest:
                v1 = comparison.TruncRoundNearest(self.v*other.v, 2*self.vlen,
                                             self.vlen-1, self.kappa)
            else:
                comparison.Trunc(v1, self.v*other.v, 2*self.vlen, self.vlen-1, self.kappa, False)
            t = v1 - c2expl
            comparison.LTZ(b, t, self.vlen+1, self.kappa)
            comparison.Trunc(v2, b*v1 + v1, self.vlen+1, 1, self.kappa, False)
            z1, z2, s1, s2, p1, p2 = (x.expand_to_vector() for x in \
                                      (self.z, other.z, self.s, other.s,
                                       self.p, other.p))
            z = z1 + z2 - self.z*other.z       # = OR(z1, z2)
            s = s1 + s2 - self.s*other.s*2     # = XOR(s1,s2)
            p = (p1 + p2 - b + self.vlen)*(1 - z)
            return sfloat(v2, p, z, s)
        else:
            return NotImplemented
    
    def __sub__(self, other):
        """ Secret floating-point subtraction.

        :param other: sfloat/float/sfix/sint/cint/regint/int """
        return self + -other
    
    def __rsub__(self, other):
        return -self + other
    __rsub__.__doc__ = __sub__.__doc__

    def __truediv__(self, other):
        """ Secret floating-point division.

        :param other: sfloat/float/sfix/sint/cint/regint/int """
        other = self.conv(other)
        v = floatingpoint.SDiv(self.v, other.v + other.z * (2**self.vlen - 1),
                               self.vlen, self.kappa, self.round_nearest)
        b = v.less_than(two_power(self.vlen-1), self.vlen + 1, self.kappa)
        overflow = v.greater_equal(two_power(self.vlen), self.vlen + 1, self.kappa)
        underflow = v.less_than(two_power(self.vlen-2), self.vlen + 1, self.kappa)
        v = (v + b * v) * (1 - overflow) * (1 - underflow) + \
            overflow * (2**self.vlen - 1) + \
            underflow * (2**(self.vlen-1)) * (1 - self.z)
        p = (1 - self.z) * (self.p - other.p - self.vlen - b + 1)
        z = self.z
        s = self.s + other.s - 2 * self.s * other.s
        sfloat.set_error(other.z)
        return sfloat(v, p, z, s)

    def __rtruediv__(self, other):
        return self.conv(other) / self
    __rtruediv__.__doc__  = __truediv__.__doc__

    @vectorize
    def __neg__(self):
        """ Secret floating-point negation. """
        return sfloat(self.v, self.p,  self.z, (1 - self.s) * (1 - self.z))

    @vectorize
    def __lt__(self, other):
        """ Secret floating-point comparison.

        :param other: sfloat/float/sfix/sint/cint/regint/int
        :return: 0/1 (sint) """
        other = self.conv(other)
        if isinstance(other, sfloat):
            z1 = self.z
            z2 = other.z
            s1 = self.s
            s2 = other.s
            a = self.p.less_than(other.p, self.plen, self.kappa)
            c = floatingpoint.EQZ(self.p - other.p, self.plen, self.kappa)
            d = ((1 - 2*self.s)*self.v).less_than((1 - 2*other.s)*other.v, self.vlen + 1, self.kappa)
            cd = c*d
            ca = c*a
            b1 = cd + a - ca
            b2 = cd + 1 + ca - c - a
            s12 = self.s*other.s
            z12 = self.z*other.z
            b = (z1 - z12)*(1 - s2) + (z2 - z12)*s1 + (1 + z12 - z1 - z2)*(s1 - s12 + (1 + s12 - s1 - s2)*b1 + s12*b2)
            return b
        else:
            return NotImplemented
    
    def __ge__(self, other):
        """ Secret floating-point comparison. """
        return 1 - (self < other)

    def __gt__(self, other):
        """ Secret floating-point comparison. """
        return self.conv(other) < self

    def __le__(self, other):
        """ Secret floating-point comparison. """
        return self.conv(other) >= self

    @vectorize
    def __eq__(self, other):
        """ Secret floating-point comparison. """
        other = self.conv(other)
        # the sign can be both ways for zeroes
        both_zero = self.z * other.z
        return floatingpoint.EQZ(self.v - other.v, self.vlen, self.kappa) * \
            floatingpoint.EQZ(self.p - other.p, self.plen, self.kappa) * \
            (1 - self.s - other.s + 2 * self.s * other.s) * \
            (1 - both_zero) + both_zero

    def __ne__(self, other):
        """ Secret floating-point comparison. """
        return 1 - (self == other)

    for op in __gt__, __le__, __ge__, __eq__, __ne__:
        op.__doc__ = __lt__.__doc__
    del op

    def log2(self):
        up = self.v.greater_than(1 << (self.vlen - 1), self.vlen, self.kappa)
        return self.p + self.vlen - 1 + up

    def round_to_int(self):
        """ Secret floating-point rounding to integer.

        :return: sint """
        direction = self.p.greater_equal(-self.vlen, self.plen, self.kappa)
        right = self.v.right_shift(-self.p - 1, self.vlen + 1, self.kappa)
        up = right.mod2m(1, self.vlen + 1, self.kappa)
        right = right.right_shift(1, self.vlen + 1, self.kappa) + up
        abs_value = direction * right
        return self.s.if_else(-abs_value, abs_value)

    def value(self):
        # Gets actual floating point value, if emulation is enabled.
        return (1 - 2*self.s.value)*(1 - self.z.value)*self.v.value/float(2**self.p.value)

    def reveal(self):
        """ Reveal secret floating-point number.

        :return: cfloat """
        return cfloat(self.v.reveal(), self.p.reveal(), self.z.reveal(), self.s.reveal())

class cfloat(object):
    """ Helper class for printing revealed sfloats. """
    __slots__ = ['v', 'p', 'z', 's', 'nan']

    def __init__(self, v, p=None, z=None, s=None, nan=0):
        """ Parameters as with :py:class:`sfloat` but public. """
        if s is None:
            parts = [cint.conv(x) for x in (v.v, v.p, v.z, v.s, v.nan)]
        else:
            parts = [cint.conv(x) for x in (v, p, z, s, nan)]
        self.v, self.p, self.z, self.s, self.nan = parts

    def print_float_plain(self):
        """ Output. """
        print_float_plain(self.v, self.p, self.z, self.s, self.nan)

sfix.float_type = sfloat

_types = {
    'c': cint,
    's': sint,
    'sg': sgf2n,
    'cg': cgf2n,
    'ci': regint,
}

def _get_type(t):
    if t in _types:
        return _types[t]
    else:
        return t

class Array(object):
    """ Array accessible by public index. """
    @classmethod
    def create_from(cls, l):
        """ Convert Python iterator to array. Basic type will be taken
        from first element, further elements must to be convertible to
        that. """
        if isinstance(l, cls):
            return l
        tmp = list(l)
        res = cls(len(tmp), type(tmp[0]))
        res.assign(tmp)
        return res

    def __init__(self, length, value_type, address=None, debug=None, alloc=True):
        """
        :param length: compile-time integer (int) or :py:obj:`None` for unknown length
        :param value_type: basic type
        :param address: if given (regint/int), the array will not be allocated """
        value_type = _get_type(value_type)
        self.address = address
        self.length = length
        self.value_type = value_type
        self.address = address
        self.address_cache = {}
        self.debug = debug
        self.creator_tape = program.curr_tape
        self.sink = None
        if alloc:
            self.alloc()

    def alloc(self):
        if self.address is None:
            self.address = self.value_type.malloc(self.length,
                                                  self.creator_tape)

    def delete(self):
        self.value_type.free(self.address)
        self.address = None

    def get_address(self, index):
        key = str(index)
        if isinstance(index, int) and self.length is not None:
            index += self.length * (index < 0)
            if index >= self.length or index < 0:
                raise IndexError('index %s, length %s' % \
                                     (str(index), str(self.length)))
        if (program.curr_block, key) not in self.address_cache:
            n = self.value_type.n_elements()
            length = self.length
            if n == 1:
                # length can be None for single-element arrays
                length = 0
            base = self.address + index * self.value_type.mem_size()
            self.address_cache[program.curr_block, key] = \
                util.untuplify([base + i * length \
                                for i in range(n)])
            if self.debug:
                library.print_ln_if(index >= self.length, 'OF:' + self.debug)
                library.print_ln_if(self.address_cache[program.curr_block, key] >= program.allocated_mem[self.value_type.reg_type], 'AOF:' + self.debug)
        return self.address_cache[program.curr_block, key]

    def get_slice(self, index):
        if index.stop is None and self.length is None:
            raise CompilerError('Cannot slice array of unknown length')
        return index.start or 0, index.stop or self.length, index.step or 1

    def __getitem__(self, index):
        """ Reading from array.

        :param index: public (regint/cint/int/slice)
        :return: array if slice is given, basic type otherwise"""
        if isinstance(index, slice):
            start, stop, step = self.get_slice(index)
            res_length = (stop - start - 1) // step + 1
            res = Array(res_length, self.value_type)
            @library.for_range(res_length)
            def f(i):
                res[i] = self[start+i*step]
            return res
        return self._load(self.get_address(index))

    def __setitem__(self, index, value):
        """ Writing to array.

        :param index: public (regint/cint/int)
        :param value: convertible for relevant basic type """
        if isinstance(index, slice):
            start, stop, step = self.get_slice(index)
            value = Array.create_from(value)
            source_index = MemValue(0)
            @library.for_range(start, stop, step)
            def f(i):
                self[i] = value[source_index]
                source_index.iadd(1)
            return
        self._store(value, self.get_address(index))

    def maybe_get(self, condition, index):
        return condition * self[condition * index]

    def maybe_set(self, condition, index, value):
        if self.sink is None:
            self.sink = self.value_type.Array(1)
        addresses = (condition.if_else(x, y) for x, y in
                     zip(util.tuplify(self.get_address(index)),
                         util.tuplify(self.sink.get_address(0))))
        self._store(value, util.untuplify(tuple(addresses)))

    # the following two are useful for compile-time lengths
    # and thus differ from the usual Python syntax
    def get_range(self, start, size):
        return [self[start + i] for i in range(size)]

    def set_range(self, start, values):
        for i, value in enumerate(values):
            self[start + i] = value

    def _load(self, address):
        return self.value_type.load_mem(address)

    def _store(self, value, address):
        self.value_type.conv(value).store_in_mem(address)

    def __len__(self):
        return self.length

    def total_size(self):
        return len(self) * self.value_type.n_elements()

    def __iter__(self):
        for i in range(self.length):
            yield self[i]

    def same_shape(self):
        return Array(self.length, self.value_type)

    def assign(self, other, base=0):
        try:
            other = other.get_vector()
        except:
            pass
        try:
            other.store_in_mem(self.get_address(base))
            if len(self) != None and util.is_constant(base):
                assert len(self) >= other.size + base
        except AttributeError:
            if isinstance(other, Array):
                @library.for_range_opt(len(other))
                def _(i):
                    self[i] = other[i]
            else:
                for i,j in enumerate(other):
                    self[i] = j
        return self

    assign_vector = assign
    assign_part_vector = assign

    def assign_all(self, value, use_threads=True, conv=True):
        """ Assign the same value to all entries.

        :param value: convertible to basic type """
        if conv:
            value = self.value_type.conv(value)
            if value.size != 1:
                raise CompilerError('cannot assign vector to all elements')
        mem_value = MemValue(value)
        self.address = MemValue.if_necessary(self.address)
        n_threads = 8 if use_threads and len(self) > 2**20 else None
        @library.for_range_multithread(n_threads, 1024, len(self))
        def f(i):
            self[i] = mem_value
        return self

    def get_vector(self, base=0, size=None):
        """ Return vector with content.

        :param base: starting point (regint/cint/int)
        :param size: length (compile-time int) """
        size = size or self.length
        return self.value_type.load_mem(self.get_address(base), size=size)

    get_part_vector = get_vector

    def get_part(self, base, size):
        return Array(size, self.value_type, self.get_address(base))

    def get(self, indices):
        return self.value_type.load_mem(
            regint.inc(len(indices), self.address, 0) + indices,
            size=len(indices))

    def get_slice_vector(self, slice):
        assert self.value_type.n_elements() == 1
        assert len(slice) <= self.total_size()
        base = regint.inc(len(slice), slice.address, 1, 1)
        inc = regint.inc(len(slice), 0, 1, 1, 1)
        addresses = slice.value_type.load_mem(base) + inc
        return self.value_type.load_mem(self.address + addresses)

    def expand_to_vector(self, index, size):
        assert self.value_type.n_elements() == 1
        address = regint(size=size)
        incint(address, regint(self.get_address(index), size=1), 0)
        return self.value_type.load_mem(address, size=size)

    def get_mem_value(self, index):
        return MemValue(self[index], self.get_address(index))

    def input_from(self, player, budget=None, raw=False):
        """ Fill with inputs from player if supported by type.

        :param player: public (regint/cint/int) """
        if raw or program.always_raw():
            input_from = self.value_type.get_raw_input_from
        else:
            input_from = self.value_type.get_input_from
        try:
            self.assign(input_from(player, size=len(self)))
        except TypeError:
            @library.for_range_opt(len(self), budget=budget)
            def _(i):
                self[i] = input_from(player)

    def __add__(self, other):
        """ Vector addition.

        :param other: vector or container of same length and type that supports operations with type of this array """
        if is_zero(other):
            return self
        assert len(self) == len(other)
        return self.get_vector() + other

    def __sub__(self, other):
        """ Vector subtraction.

        :param other: vector or container of same length and type that supports operations with type of this array """
        assert len(self) == len(other)
        return self.get_vector() - other

    def __mul__(self, value):
        """ Vector multiplication.

        :param other: vector or container of same length and type that supports operations with type of this array """
        return self.get_vector() * value

    def __truediv__(self, value):
        """ Vector division.

        :param other: vector or container of same length and type that supports operations with type of this array """
        return self.get_vector() / value

    def __pow__(self, value):
        """ Vector power-of computation.

        :param other: compile-time integer (int) """
        return self.get_vector() ** value

    __radd__ = __add__
    __rmul__ = __mul__

    def shuffle(self):
        """ Insecure shuffle in place. """
        if self.value_type == regint:
            self.assign(self.get_vector().shuffle())
        else:
            @library.for_range(len(self))
            def _(i):
                j = regint.get_random(64) % (len(self) - i)
                tmp = self[i]
                self[i] = self[i + j]
                self[i + j] = tmp

    def reveal(self):
        """ Reveal the whole array.

        :returns: Array of relevant clear type. """
        return Array.create_from(x.reveal() for x in self)

    def reveal_list(self):
        """ Reveal as list. """
        return list(self.get_vector().reveal())

    reveal_nested = reveal_list

    def sort(self, n_threads=None):
        """
        Sort in place using Batchers' odd-even merge mergesort
        with complexity :math:`O(n (\log n)^2)`.

        :param n_threads: number of threads to use (single thread by
          default)
        """
        library.loopy_odd_even_merge_sort(self, n_threads=n_threads)

    def __str__(self):
        return '%s array of length %s at %s' % (self.value_type, len(self),
                                                self.address)

sint.dynamic_array = Array
sgf2n.dynamic_array = Array


class SubMultiArray(object):
    """ Multidimensional array functionality. """
    def __init__(self, sizes, value_type, address, index, debug=None):
        """ Do not call this, use :py:class:`MultiArray` instead. """
        self.sizes = tuple(sizes)
        self.value_type = _get_type(value_type)
        if address is not None:
            self.address = address + index * self.total_size()
        else:
            self.address = None
        self.sub_cache = {}
        self.debug = debug
        if debug:
            library.print_ln_if(self.address + reduce(operator.mul, self.sizes) * self.value_type.n_elements() > program.allocated_mem[self.value_type.reg_type], 'AOF%d:' % len(self.sizes) + self.debug)

    def __getitem__(self, index):
        """ Part access.

        :param index: public (regint/cint/int)
        :return: :py:class:`Array` if one-dimensional, :py:class:`SubMultiArray` otherwise"""
        if util.is_constant(index) and index >= self.sizes[0]:
            raise StopIteration
        key = program.curr_block, str(index)
        if key not in self.sub_cache:
            if self.debug:
                library.print_ln_if(index >= self.sizes[0], \
                                    'OF%d:' % len(self.sizes) + self.debug)
            if len(self.sizes) == 2:
                self.sub_cache[key] = \
                        Array(self.sizes[1], self.value_type, \
                              self.address + index * self.sizes[1] *
                              self.value_type.n_elements(), \
                              debug=self.debug)
            else:
                self.sub_cache[key] = \
                        SubMultiArray(self.sizes[1:], self.value_type, \
                                      self.address, index, debug=self.debug)
        return self.sub_cache[key]

    def __setitem__(self, index, other):
        """ Part assignment.

        :param index: public (regint/cint/int)
        :param other: container of matching size and type """
        self[index].assign(other)

    def __len__(self):
        """ Size of top dimension. """
        return self.sizes[0]

    def assign_all(self, value):
        """ Assign the same value to all entries.

        :param value: convertible to relevant basic type """
        @library.for_range(self.sizes[0])
        def f(i):
            self[i].assign_all(value)
        return self

    def total_size(self):
        return reduce(operator.mul, self.sizes) * self.value_type.n_elements()

    def get_vector(self, base=0, size=None):
        """ Return vector with content. Not implemented for floating-point.

        :param base: public (regint/cint/int)
        :param size: compile-time (int) """
        assert self.value_type.n_elements() == 1
        size = size or self.total_size()
        return self.value_type.load_mem(self.address + base, size=size)

    def assign_vector(self, vector, base=0):
        """ Assign vector to content. Not implemented for floating-point.

        :param vector: vector of matching size convertible to relevant basic type
        :param base: compile-time (int) """
        assert self.value_type.n_elements() == 1
        assert vector.size <= self.total_size()
        vector.store_in_mem(self.address + base)

    def assign(self, other):
        """ Assign container to content. Not implemented for floating-point.

        :param other: container of matching size and type """
        if self.value_type.n_elements() > 1:
            assert self.sizes == other.sizes
        self.assign_vector(other.get_vector())

    def get_part_vector(self, base=0, size=None):
        assert self.value_type.n_elements() == 1
        part_size = reduce(operator.mul, self.sizes[1:])
        size = (size or 1) * part_size
        assert size <= self.total_size()
        return self.value_type.load_mem(self.address + base * part_size,
                                        size=size)

    def assign_part_vector(self, vector, base=0):
        assert self.value_type.n_elements() == 1
        part_size = reduce(operator.mul, self.sizes[1:])
        assert vector.size <= self.total_size()
        vector.store_in_mem(self.address + base * part_size)

    def get_slice_vector(self, slice):
        assert self.value_type.n_elements() == 1
        part_size = reduce(operator.mul, self.sizes[1:])
        assert len(slice) * part_size <= self.total_size()
        base = regint.inc(len(slice) * part_size, slice.address, 1, part_size)
        inc = regint.inc(len(slice) * part_size, 0, 1, 1, part_size)
        addresses = slice.value_type.load_mem(base) * part_size + inc
        return self.value_type.load_mem(self.address + addresses)

    def get_addresses(self, *indices):
        assert self.value_type.n_elements() == 1
        assert len(indices) == len(self.sizes)
        size = 1
        base = 0
        skip = 1
        has_glob = False
        last_was_glob = False
        for i, x in enumerate(indices):
            part_size = reduce(operator.mul, (1,) + self.sizes[i + 1:])
            if x is None:
                assert not has_glob or last_was_glob
                has_glob = True
                size *= self.sizes[i]
                skip = part_size
            else:
                base += x * part_size
            last_was_glob = x is None
        res = regint.inc(size, self.address + base, skip)
        return res

    def get_vector_by_indices(self, *indices):
        addresses = self.get_addresses(*indices)
        return self.value_type.load_mem(addresses)

    def assign_vector_by_indices(self, vector, *indices):
        addresses = self.get_addresses(*indices)
        vector.store_in_mem(addresses)

    def same_shape(self):
        """ :return: new multidimensional array with same shape and basic type """
        return MultiArray(self.sizes, self.value_type)

    def get_part(self, start, size):
        return MultiArray([size] + list(self.sizes[1:]), self.value_type,
                          address=self[start].address)

    def input_from(self, player, budget=None, raw=False):
        """ Fill with inputs from player if supported by type.

        :param player: public (regint/cint/int) """
        budget = budget or Tape.Register.maximum_size
        if (self.total_size() < budget) and \
           self.value_type.n_elements() == 1:
            if raw or program.always_raw():
                input_from = self.value_type.get_raw_input_from
            else:
                input_from = self.value_type.get_input_from
            self.assign_vector(input_from(player, size=self.total_size()))
        else:
            @library.for_range_opt(self.sizes[0],
                                   budget=budget / self[0].total_size())
            def _(i):
                self[i].input_from(player, budget=budget, raw=raw)

    def schur(self, other):
        """ Element-wise product.

        :param other: container of matching size and type
        :return: container of same shape and type as :py:obj:`self` """
        assert self.sizes == other.sizes
        if len(self.sizes) == 2:
            res = Matrix(self.sizes[0], self.sizes[1], self.value_type)
        else:
            res = MultiArray(self.sizes, self.value_type)
        res.assign_vector(self.get_vector() * other.get_vector())
        return res

    def __add__(self, other):
        """ Element-wise addition.

        :param other: container of matching size and type
        :return: container of same shape and type as :py:obj:`self` """
        if is_zero(other):
            return self
        assert self.sizes == other.sizes
        if len(self.sizes) == 2:
            res = Matrix(self.sizes[0], self.sizes[1], self.value_type)
        else:
            res = MultiArray(self.sizes, self.value_type)
        res.assign_vector(self.get_vector() + other.get_vector())
        return res

    __radd__ = __add__

    def iadd(self, other):
        """ Element-wise addition in place.

        :param other: container of matching size and type """
        assert self.sizes == other.sizes
        self.assign_vector(self.get_vector() + other.get_vector())

    def __mul__(self, other):
        """ Matrix-matrix and matrix-vector multiplication.

        :param self: two-dimensional
        :param other: Matrix or Array of matching size and type """
        return self.mul(other)

    def mul(self, other, res_params=None):
        assert len(self.sizes) == 2
        if isinstance(other, Array):
            assert len(other) == self.sizes[1]
            if self.value_type.n_elements() == 1:
                matrix = Matrix(len(other), 1, other.value_type, \
                                address=other.address)
                res = self * matrix
                return Array(res.sizes[0], res.value_type, address=res.address)
            else:
                matrix = Matrix(len(other), 1, other.value_type)
                for i, x in enumerate(other):
                    matrix[i][0] = x
                res = self * matrix
                library.break_point()
                return Array.create_from(x[0] for x in res)
        elif isinstance(other, SubMultiArray):
            assert len(other.sizes) == 2
            assert other.sizes[0] == self.sizes[1]
            if res_params is not None:
                class t(self.value_type):
                    pass
                t.params = res_params
            else:
                t = self.value_type
            res_matrix = Matrix(self.sizes[0], other.sizes[1], t)
            try:
                try:
                    res_matrix.assign_vector(self.direct_mul(other))
                except AttributeError:
                    if max(res_matrix.sizes) > 1000:
                        raise AttributeError()
                    A = self.get_vector()
                    B = other.get_vector()
                    res_matrix.assign_vector(
                        self.value_type.matrix_mul(A, B, self.sizes[1],
                                                   res_params))
            except (AttributeError, AssertionError):
                # fallback for sfloat etc.
                @library.for_range_opt(self.sizes[0])
                def _(i):
                    try:
                        res_matrix[i] = self.value_type.row_matrix_mul(
                            self[i], other, res_params)
                    except AttributeError:
                        # fallback for binary circuits
                        @library.for_range(other.sizes[1])
                        def _(j):
                            res_matrix[i][j] = 0
                            @library.for_range(self.sizes[1])
                            def _(k):
                                res_matrix[i][j] += self[i][k] * other[k][j]
            return res_matrix
        else:
            raise NotImplementedError

    def direct_mul(self, other, reduce=True, indices=None):
        """ Matrix multiplication in the virtual machine.

        :param self: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param other: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param indices: 4-tuple of :py:class:`regint` vectors for index selection (default is complete multiplication)
        :return: Matrix as vector of relevant type (row-major)

        The following executes a matrix multiplication selecting every third row
        of :py:obj:`A`::

            A = sfix.Matrix(7, 4)
            B = sfix.Matrix(4, 5)
            C = sfix.Matrix(3, 5)
            C.assign_vector(A.direct_mul(B, indices=(regint.inc(3, 0, 3),
                                                     regint.inc(4),
                                                     regint.inc(4),
                                                     regint.inc(5)))
        """
        assert len(self.sizes) == 2
        assert len(other.sizes) == 2
        assert self.sizes[1] == other.sizes[0]
        return self.value_type.direct_matrix_mul(self.address, other.address,
                                                 self.sizes[0], *other.sizes,
                                                 reduce=reduce, indices=indices)

    def direct_mul_trans(self, other, reduce=True, indices=None):
        """
        Matrix multiplication with the transpose of :py:obj:`other`
        in the virtual machine.

        :param self: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param other: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param indices: 4-tuple of :py:class:`regint` vectors for index selection (default is complete multiplication)
        :return: Matrix as vector of relevant type (row-major)

        """
        assert len(self.sizes) == 2
        assert len(other.sizes) == 2
        if indices is None:
            assert self.sizes[1] == other.sizes[1]
            indices = [regint.inc(i) for i in self.sizes + other.sizes[::-1]]
        assert len(indices[1]) == len(indices[2])
        indices = list(indices)
        indices[3] *= other.sizes[1]
        return self.value_type.direct_matrix_mul(
            self.address, other.address, None, self.sizes[1], 1,
            reduce=reduce, indices=indices)

    def direct_trans_mul(self, other, reduce=True, indices=None):
        """
        Matrix multiplication with the transpose of :py:obj:`self`
        in the virtual machine.

        :param self: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param other: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param indices: 4-tuple of :py:class:`regint` vectors for index selection (default is complete multiplication)
        :return: Matrix as vector of relevant type (row-major)

        """
        assert len(self.sizes) == 2
        assert len(other.sizes) == 2
        if indices is None:
            assert self.sizes[0] == other.sizes[0]
            indices = [regint.inc(i) for i in self.sizes[::-1] + other.sizes]
        assert len(indices[1]) == len(indices[2])
        indices = list(indices)
        indices[1] *= self.sizes[1]
        return self.value_type.direct_matrix_mul(
            self.address, other.address, None, 1, other.sizes[1],
            reduce=reduce, indices=indices)

    def direct_mul_to_matrix(self, other):
        """ Matrix multiplication in the virtual machine.

        :param self: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :param other: :py:class:`Matrix` / 2-dimensional :py:class:`MultiArray`
        :returns: :py:obj:`Matrix`

        """
        res = self.value_type.Matrix(self.sizes[0], other.sizes[1])
        res.assign_vector(self.direct_mul(other))
        return res

    def budget_mul(self, other, n_rows, row, n_columns, column, reduce=True,
                   res=None):
        assert len(self.sizes) == 2
        assert len(other.sizes) == 2
        if res is None:
            if reduce:
                res_matrix = Matrix(n_rows, n_columns, self.value_type)
            else:
                res_matrix = Matrix(n_rows, n_columns, \
                                    self.value_type.unreduced_type)
        else:
            res_matrix = res
        @library.for_range_opt(n_rows)
        def _(i):
            @library.for_range_opt(n_columns)
            def _(j):
                col = column(other, j)
                r = row(self, i)
                if reduce:
                    res_matrix[i][j] = self.value_type.dot_product(r, col)
                else:
                    entry = self.value_type.unreduced_dot_product(r, col)
                    res_matrix[i][j] = entry
        return res_matrix

    def plain_mul(self, other, res=None):
        """ Alternative matrix multiplication.

        :param self: two-dimensional
        :param other: two-dimensional container of matching type and size """
        assert other.sizes[0] == self.sizes[1]
        return self.budget_mul(other, self.sizes[0], lambda x, i: x[i], \
                               other.sizes[1], \
                               lambda x, j: [x[k][j] for k in range(len(x))],
                               res=res)

    def mul_trans(self, other):
        """ Matrix multiplication with transpose of :py:obj:`other`.

        :param self: two-dimensional
        :param other: two-dimensional container of matching type and size """
        assert other.sizes[1] == self.sizes[1]
        return self.budget_mul(other, self.sizes[0], lambda x, i: x[i], \
                               other.sizes[0], lambda x, j: x[j])

    def trans_mul(self, other, reduce=True, res=None):
        """ Matrix multiplication with transpose of :py:obj:`self`

        :param self: two-dimensional
        :param other: two-dimensional container of matching type and size """
        assert other.sizes[0] == self.sizes[0]
        return self.budget_mul(other, self.sizes[1], \
                               lambda x, j: [x[k][j] for k in range(len(x))], \
                               other.sizes[1], \
                               lambda x, j: [x[k][j] for k in range(len(x))],
                               reduce=reduce, res=res)

    def parallel_mul(self, other):
        assert self.sizes[1] == other.sizes[0]
        assert len(self.sizes) == 2
        assert len(other.sizes) == 2
        assert self.value_type.n_elements() == 1
        n = self.sizes[0] * other.sizes[1]
        a = []
        b = []
        for i in range(self.sizes[1]):
            addresses = regint(size=n)
            incint(addresses, regint(self.address + i), self.sizes[1],
                   other.sizes[1], n)
            a.append(self.value_type.load_mem(addresses, size=n))
            addresses = regint(size=n)
            incint(addresses, regint(other.address + i * other.sizes[1]), 1,
                   1, other.sizes[1])
            b.append(self.value_type.load_mem(addresses, size=n))
        res = self.value_type.dot_product(a, b)
        return res

    def transpose(self):
        """ Matrix transpose.

        :param self: two-dimensional """
        assert len(self.sizes) == 2
        res = Matrix(self.sizes[1], self.sizes[0], self.value_type)
        library.break_point()
        @library.for_range_opt(self.sizes[1])
        def _(i):
            @library.for_range_opt(self.sizes[0])
            def _(j):
                res[i][j] = self[j][i]
        library.break_point()
        return res

    def reveal_list(self):
        """ Reveal as list. """
        return list(self.get_vector().reveal())

    def reveal_nested(self):
        """ Reveal as nested list. """
        flat = iter(self.get_vector().reveal())
        res = []
        def f(sizes):
            if len(sizes) == 1:
                return [next(flat) for i in range(sizes[0])]
            else:
                return [f(sizes[1:]) for i in range(sizes[0])]
        return f(self.sizes)

    def __str__(self):
        return '%s multi-array of lengths %s at %s' % (self.value_type,
                                                       self.sizes, self.address)

class MultiArray(SubMultiArray):
    """ Multidimensional array. """
    def __init__(self, sizes, value_type, debug=None, address=None, alloc=True):
        """
        :param sizes: shape (compile-time list of integers)
        :param value_type: basic type of entries
        """
        if isinstance(address, Array):
            self.array = address
        else:
            self.array = Array(reduce(operator.mul, sizes), \
                               value_type, address=address, alloc=alloc)
        SubMultiArray.__init__(self, sizes, value_type, self.array.address, 0, \
                               debug=debug)
        if len(sizes) < 2:
            raise CompilerError('Use Array')

    @property
    def address(self):
        return self.array.address

    @address.setter
    def address(self, value):
        self.array.address = value

    def alloc(self):
        self.array.alloc()

    def delete(self):
        self.array.delete()

class Matrix(MultiArray):
    """ Matrix. """
    def __init__(self, rows, columns, value_type, debug=None, address=None):
        """
        :param rows: compile-time (int)
        :param columns: compile-time (int)
        :param value_type: basic type of entries
        """
        MultiArray.__init__(self, [rows, columns], value_type, debug=debug, \
                            address=address)

    def set_column(self, index, vector):
        assert self.value_type.n_elements() == 1
        addresses = regint.inc(self.sizes[0], self.address + index,
                               self.sizes[1])
        vector.store_in_mem(addresses)

class VectorArray(object):
    def __init__(self, length, value_type, vector_size, address=None):
        self.array = Array(length * vector_size, value_type, address)
        self.vector_size = vector_size
        self.value_type = value_type

    def __getitem__(self, index):
        return self.value_type.load_mem(self.array.address + \
                                        index * self.vector_size,
                                        size=self.vector_size)

    def __setitem__(self, index, value):
        if value.size != self.vector_size:
            raise CompilerError('vector size mismatch')
        value.store_in_mem(self.array.address + index * self.vector_size)

class _mem(_number):
    __add__ = lambda self,other: self.read() + other
    __sub__ = lambda self,other: self.read() - other
    __mul__ = lambda self,other: self.read() * other
    __truediv__ = lambda self,other: self.read() / other
    __floordiv__ = lambda self,other: self.read() // other
    __mod__ = lambda self,other: self.read() % other
    __pow__ = lambda self,other: self.read() ** other
    __neg__ = lambda self,other: -self.read()
    __lt__ = lambda self,other: self.read() < other
    __gt__ = lambda self,other: self.read() > other
    __le__ = lambda self,other: self.read() <= other
    __ge__ = lambda self,other: self.read() >= other
    __eq__ = lambda self,other: self.read() == other
    __ne__ = lambda self,other: self.read() != other
    __and__ = lambda self,other: self.read() & other
    __xor__ = lambda self,other: self.read() ^ other
    __or__ = lambda self,other: self.read() | other
    __lshift__ = lambda self,other: self.read() << other
    __rshift__ = lambda self,other: self.read() >> other

    __radd__ = lambda self,other: other + self.read()
    __rsub__ = lambda self,other: other - self.read()
    __rmul__ = lambda self,other: other * self.read()
    __rtruediv__ = lambda self,other: other / self.read()
    __rfloordiv__ = lambda self,other: other // self.read()
    __rmod__ = lambda self,other: other % self.read()
    __rand__ = lambda self,other: other & self.read()
    __rxor__ = lambda self,other: other ^ self.read()
    __ror__ = lambda self,other: other | self.read()

    __iadd__ = lambda self,other: self.write(self.read() + other)
    __isub__ = lambda self,other: self.write(self.read() - other)
    __imul__ = lambda self,other: self.write(self.read() * other)
    __itruediv__ = lambda self,other: self.write(self.read() / other)
    __ifloordiv__ = lambda self,other: self.write(self.read() // other)
    __imod__ = lambda self,other: self.write(self.read() % other)
    __ipow__ = lambda self,other: self.write(self.read() ** other)
    __iand__ = lambda self,other: self.write(self.read() & other)
    __ixor__ = lambda self,other: self.write(self.read() ^ other)
    __ior__ = lambda self,other: self.write(self.read() | other)
    __ilshift__ = lambda self,other: self.write(self.read() << other)
    __irshift__ = lambda self,other: self.write(self.read() >> other)

    iadd = __iadd__
    isub = __isub__
    imul = __imul__
    itruediv = __itruediv__
    ifloordiv = __ifloordiv__
    imod = __imod__
    ipow = __ipow__
    iand = __iand__
    ixor = __ixor__
    ior = __ior__
    ilshift = __ilshift__
    irshift = __irshift__

    store_in_mem = lambda self,address: self.read().store_in_mem(address)

class MemValue(_mem):
    """ Single value in memory. This is useful to transfer information
    between threads. Operations are automatically read
    from memory if required, this means you can use any operation with
    :py:class:`MemValue` objects as if they were a basic type. """
    __slots__ = ['last_write_block', 'reg_type', 'register', 'address', 'deleted']

    @classmethod
    def if_necessary(cls, value):
        if util.is_constant_float(value):
            return value
        else:
            return cls(value)

    def __init__(self, value, address=None):
        """ :param value: basic type or int (will be converted to regint) """
        self.last_write_block = None
        if isinstance(value, int):
            self.value_type = regint
            value = regint(value)
        elif isinstance(value, MemValue):
            self.value_type = value.value_type
        else:
            self.value_type = type(value)
        self.deleted = False
        if address is None:
            self.address = self.value_type.malloc(value.size)
            self.size = value.size
            self.write(value)
        else:
            self.address = address
            self.size = 1

    def delete(self):
        self.value_type.free(self.address)
        self.deleted = True

    def check(self):
        if self.deleted:
            raise CompilerError('MemValue deleted')

    def read(self):
        """ Read value.

        :return: relevant basic type instance """
        self.check()
        if program.curr_block != self.last_write_block:
            self.register = self.value_type.load_mem(self.address)
            self.last_write_block = program.curr_block
        return self.register

    def write(self, value):
        """ Write value.

        :param value: convertible to relevant basic type """
        self.check()
        if isinstance(value, MemValue):
            value = value.read()
        value = self.value_type.conv(value)
        if value.size != self.size:
            raise CompilerError('size mismatch')
        self.register = value
        if not isinstance(self.register, self.value_type):
            raise CompilerError('Mismatch in register type, cannot write \
                %s to %s' % (type(self.register), self.value_type))
        self.register.store_in_mem(self.address)
        self.last_write_block = program.curr_block
        return self

    def reveal(self):
        """ Reveal value.

        :return: relevant clear type """
        return self.read().reveal()

    less_than = lambda self,other,bit_length=None,security=None: \
        self.read().less_than(other,bit_length,security)
    greater_than = lambda self,other,bit_length=None,security=None: \
        self.read().greater_than(other,bit_length,security)
    less_equal = lambda self,other,bit_length=None,security=None: \
        self.read().less_equal(other,bit_length,security)
    greater_equal = lambda self,other,bit_length=None,security=None: \
        self.read().greater_equal(other,bit_length,security)
    equal = lambda self,other,bit_length=None,security=None: \
        self.read().equal(other,bit_length,security)
    not_equal = lambda self,other,bit_length=None,security=None: \
        self.read().not_equal(other,bit_length,security)

    pow2 = lambda self,*args,**kwargs: self.read().pow2(*args, **kwargs)
    mod2m = lambda self,*args,**kwargs: self.read().mod2m(*args, **kwargs)
    right_shift = lambda self,*args,**kwargs: self.read().right_shift(*args, **kwargs)

    bit_decompose = lambda self,*args,**kwargs: self.read().bit_decompose(*args, **kwargs)

    if_else = lambda self,*args,**kwargs: self.read().if_else(*args, **kwargs)
    bit_and = lambda self,other: self.read().bit_and(other)

    def expand_to_vector(self, size=None):
        if program.curr_block == self.last_write_block:
            return self.read().expand_to_vector(size)
        else:
            if size is None:
                size = get_global_vector_size()
            addresses = regint.inc(size, self.address, 0)
            return self.value_type.load_mem(addresses)

    def __repr__(self):
        return 'MemValue(%s,%d)' % (self.value_type, self.address)


class MemFloat(MemValue):
    def __init__(self, *args):
        super().__init__(sfloat(*args))

    def write(self, *args):
        value = sfloat(*args)
        super().write(value)

class MemFix(MemValue):
    def __init__(self, *args):
        arg_type = type(*args)
        if arg_type == sfix:
            value = sfix(*args)
        elif arg_type == cfix:
            value = cfix(*args)
        else:
            raise CompilerError('MemFix init argument error')
        super().__init__(value)

    def write(self, *args):
        super().write(self.value_type(*args))

def getNamedTupleType(*names):
    class NamedTuple(object):
        class NamedTupleArray(object):
            def __init__(self, size, t):
                from . import types
                self.arrays = [types.Array(size, t) for i in range(len(names))]
            def __getitem__(self, index):
                return NamedTuple(array[index] for array in self.arrays)
            def __setitem__(self, index, item):
                for array,value in zip(self.arrays, item):
                    array[index] = value
        @classmethod
        def get_array(cls, size, t):
            return cls.NamedTupleArray(size, t)
        def __init__(self, *args):
            if len(args) == 1:
                args = args[0]
            for name, value in zip(names, args):
                self.__dict__[name] = value
        def __iter__(self):
            for name in names:
                yield self.__dict__[name]
        def __add__(self, other):
            return NamedTuple(i + j for i,j in zip(self, other))
        def __sub__(self, other):
            return NamedTuple(i - j for i,j in zip(self, other))
        def __xor__(self, other):
            return NamedTuple(i ^ j for i,j in zip(self, other))
        def __mul__(self, other):
            return NamedTuple(other * i for i in self)
        __rmul__ = __mul__
        __rxor__ = __xor__
        def reveal(self):
            return self.__type__(x.reveal() for x in self)
    return NamedTuple

from . import library
