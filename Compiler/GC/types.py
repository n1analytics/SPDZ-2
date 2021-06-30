"""
This modules contains basic types for binary circuits. The
fixed-length types obtained by :py:obj:`get_type(n)` are the preferred
way of using them, and in some cases required in connection with
container types.
"""

from Compiler.types import MemValue, read_mem_value, regint, Array, cint
from Compiler.types import _bitint, _number, _fix, _structure, _register, _bit, _vec, sint
from Compiler.program import Tape, Program
from Compiler.exceptions import *
from Compiler import util, oram, floatingpoint, library
from Compiler import instructions_base
import Compiler.GC.instructions as inst
import operator
import math
from functools import reduce

class bits(Tape.Register, _structure, _bit):
    """ Base class for binary registers. """
    n = 40
    unit = 64
    PreOp = staticmethod(floatingpoint.PreOpN)
    decomposed = None
    @staticmethod
    def PreOR(l):
        return [1 - x for x in \
                floatingpoint.PreOpN(operator.mul, \
                                     [1 - x for x in l])]
    @classmethod
    def get_type(cls, length):
        """ Returns a fixed-length type. """
        if length == 1:
            return cls.bit_type
        if length not in cls.types:
            class bitsn(cls):
                n = length
            cls.types[length] = bitsn
            bitsn.__name__ = cls.__name__ + str(length)
        return cls.types[length]
    @classmethod
    def conv(cls, other):
        if isinstance(other, cls):
            return other
        elif isinstance(other, MemValue):
            return cls.conv(other.read())
        else:
            res = cls()
            res.load_other(other)
            return res
    hard_conv = conv
    @classmethod
    def compose(cls, items, bit_length=1):
        return cls.bit_compose(sum([util.bit_decompose(item, bit_length) for item in items], []))
    @classmethod
    def bit_compose(cls, bits):
        bits = list(bits)
        if len(bits) == 1:
            return bits[0]
        bits = list(bits)
        for i in range(len(bits)):
            if util.is_constant(bits[i]):
                bits[i] = sbit(bits[i])
        res = cls.new(n=len(bits))
        if len(bits) <= cls.unit:
            cls.bitcom(res, *(sbit.conv(bit) for bit in bits))
        else:
            n_bak = bits[0].n
            bits[0].n = 1
            res = cls.trans(bits)[0]
            bits[0].n = n_bak
        res.decomposed = bits
        return res
    def bit_decompose(self, bit_length=None):
        n = bit_length or self.n
        suffix = [0] * (n - self.n)
        if n == 1 and self.n == 1:
            return [self]
        n = min(n, self.n)
        if self.decomposed is None or len(self.decomposed) < n:
            if n <= self.unit:
                res = [self.bit_type() for i in range(n)]
                self.bitdec(self, *res)
            else:
                res = self.bit_type.trans([self])
            self.decomposed = res
            return res + suffix
        else:
            return self.decomposed[:n] + suffix
    @staticmethod
    def bit_decompose_clear(a, n_bits):
        res = [cbits.get_type(a.size)() for i in range(n_bits)]
        cbits.conv_cint_vec(a, *res)
        return res
    @classmethod
    def malloc(cls, size, creator_tape=None):
        return Program.prog.malloc(size, cls, creator_tape=creator_tape)
    @staticmethod
    def n_elements():
        return 1
    @classmethod
    def mem_size(cls):
        return math.ceil(cls.n / cls.unit)
    @classmethod
    def load_mem(cls, address, mem_type=None, size=None):
        if size not in (None, 1):
            v = [cls.load_mem(address + i) for i in range(size)]
            return cls.vec(v)
        res = cls()
        if mem_type == 'sd':
            return cls.load_dynamic_mem(address)
        else:
            is_constant = util.is_constant(address)
            if isinstance(address, _register):
                address = _register._expand_address(address,
                        instructions_base.get_global_vector_size())
            for i in range(res.size):
                cls.load_inst[is_constant](res[i], address + i)
            return res
    def store_in_mem(self, address):
        is_int = isinstance(address, int)
        if isinstance(address, _register):
            address = _register._expand_address(address, self.n)
        self.store_inst[is_int](self, address)
    def __init__(self, value=None, n=None, size=None):
        if size != 1 and size is not None:
            raise Exception('invalid size for bit type: %s' % size)
        self.n = n or self.n
        size = math.ceil(self.n / self.unit) if self.n != None else None
        Tape.Register.__init__(self, self.reg_type, Program.prog.curr_tape,
                               size=size)
        if value is not None:
            self.load_other(value)
    def copy(self):
        return type(self)(n=instructions_base.get_global_vector_size())
    def set_length(self, n):
        if n > self.n:
            raise Exception('too long: %d/%d' % (n, self.n))
        self.n = n
    def set_size(self, size):
        pass
    def load_other(self, other):
        if isinstance(other, cint):
            assert(self.n == other.size)
            self.conv_regint_by_bit(self.n, self, other.to_regint(1))
        elif isinstance(other, int):
            self.set_length(self.n or util.int_len(other))
            self.load_int(other)
        elif isinstance(other, regint):
            assert(other.size == math.ceil(self.n / self.unit))
            for i, (x, y) in enumerate(zip(self, other)):
                self.conv_regint(min(self.unit, self.n - i * self.unit), x, y)
        elif (isinstance(self, type(other)) or isinstance(other, type(self))) \
             and self.n == other.n:
            for i in range(math.ceil(self.n / self.unit)):
                self.mov(self[i], other[i])
        elif isinstance(other, sint):
            self.mov(self, sbitvec(other, self.n).elements()[0])
        else:
            try:
                bits = other.bit_decompose()
                bits = bits[:self.n] + [sbit(0)] * (self.n - len(bits))
                other = self.bit_compose(bits)
                self.load_other(other)
            except:
                raise CompilerError('cannot convert %s/%s from %s to %s' % \
                                    (str(other), repr(other), type(other), type(self)))
    def long_one(self):
        return 2**self.n - 1 if self.n != None else None
    def __repr__(self):
        if self.n != None:
            suffix = '%d' % self.n
            if type(self).n != None and type(self).n != self.n:
                suffix += '/%d' % type(self).n
        else:
            suffix = 'undef'
        return '%s(%s)' % (super(bits, self).__repr__(), suffix)
    __str__ = __repr__
    def _new_by_number(self, i, size=1):
        assert(size == 1)
        n = min(self.unit, self.n - (i - self.i) * self.unit)
        res = self.get_type(n)()
        res.i = i
        res.program = self.program
        return res

class cbits(bits):
    """ Clear bits register. Helper type with limited functionality. """
    max_length = 64
    reg_type = 'cb'
    is_clear = True
    load_inst = (None, inst.ldmcb)
    store_inst = (None, inst.stmcb)
    bitdec = inst.bitdecc
    conv_regint = staticmethod(lambda n, x, y: inst.convcint(x, y))
    conv_cint_vec = inst.convcintvec
    @classmethod
    def bit_compose(cls, bits):
        return sum(bit << i for i, bit in enumerate(bits))
    @classmethod
    def conv_regint_by_bit(cls, n, res, other):
        assert n == res.n
        assert n == other.size
        cls.conv_cint_vec(cint(other, size=other.size), res)
    types = {}
    def load_int(self, value):
        self.load_other(regint(value))
    def store_in_dynamic_mem(self, address):
        inst.stmsdci(self, cbits.conv(address))
    def clear_op(self, other, c_inst, ci_inst, op):
        if isinstance(other, cbits):
            res = cbits(n=max(self.n, other.n))
            c_inst(res, self, other)
            return res
        else:
            if util.is_constant(other):
                if other >= 2**31 or other < -2**31:
                    return op(self, cbits(other))
                res = cbits(n=max(self.n, len(bin(other)) - 2))
                ci_inst(res, self, other)
                return res
            else:
                return op(self, cbits(other))
    __add__ = lambda self, other: \
              self.clear_op(other, inst.addcb, inst.addcbi, operator.add)
    __sub__ = lambda self, other: \
              self.clear_op(-other, inst.addcb, inst.addcbi, operator.add)
    def __xor__(self, other):
        if isinstance(other, (sbits, sbitvec)):
            return NotImplemented
        else:
            self.clear_op(other, inst.xorcb, inst.xorcbi, operator.xor)
    __radd__ = __add__
    __rxor__ = __xor__
    def __mul__(self, other):
        if isinstance(other, cbits):
            return NotImplemented
        else:
            try:
                res = cbits(n=min(self.max_length, self.n+util.int_len(other)))
                inst.mulcbi(res, self, other)
                return res
            except TypeError:
                return NotImplemented
    def __rshift__(self, other):
        res = cbits(n=self.n-other)
        inst.shrcbi(res, self, other)
        return res
    def __lshift__(self, other):
        res = cbits(n=self.n+other)
        inst.shlcbi(res, self, other)
        return res
    def print_reg(self, desc=''):
        inst.print_regb(self, desc)
    def print_reg_plain(self):
        inst.print_reg_signed(self.n, self)
    output = print_reg_plain
    def print_if(self, string):
        inst.cond_print_strb(self, string)
    def reveal(self):
        return self
    def to_regint(self, dest=None):
        if dest is None:
            dest = regint()
        if self.n > 64:
            raise CompilerError('too many bits')
        inst.convcbit(dest, self)
        return dest
    def to_regint_by_bit(self):
        if self.n != None:
            res = regint(size=self.n)
        else:
            res = regint()
        inst.convcbitvec(self.n, res, self)
        return res

class sbits(bits):
    """
    Secret bits register. This type supports basic bit-wise operations::

        sb32 = sbits.get_type(32)
        a = sb32(3)
        b = sb32(5)
        print_ln('XOR: %s', (a ^ b).reveal())
        print_ln('AND: %s', (a & b).reveal())
        print_ln('NOT: %s', (~a).reveal())

    This will output the following::

        XOR: 6
        AND: 1
        NOT: -4

    Instances can be also be initalized from :py:obj:`~Compiler.types.regint`
    and :py:obj:`~Compiler.types.sint`.
    """
    max_length = 64
    reg_type = 'sb'
    is_clear = False
    clear_type = cbits
    default_type = cbits
    load_inst = (inst.ldmsbi, inst.ldmsb)
    store_inst = (inst.stmsbi, inst.stmsb)
    bitdec = inst.bitdecs
    bitcom = inst.bitcoms
    conv_regint = inst.convsint
    @classmethod
    def conv_regint_by_bit(cls, n, res, other):
        tmp = cbits.get_type(n)()
        tmp.conv_regint_by_bit(n, tmp, other)
        res.load_other(tmp)
    mov = inst.movsb
    types = {}
    def __init__(self, *args, **kwargs):
        bits.__init__(self, *args, **kwargs)
    @staticmethod
    def new(value=None, n=None):
        if n == 1:
            return sbit(value)
        else:
            return sbits.get_type(n)(value)
    @staticmethod
    def get_random_bit():
        res = sbit()
        inst.bitb(res)
        return res
    @classmethod
    def get_input_from(cls, player, n_bits=None):
        """ Secret input from :py:obj:`player`.

        :param: player (int)
        """
        if n_bits is None:
            n_bits = cls.n
        res = cls()
        inst.inputb(player, n_bits, 0, res)
        return res
    # compatiblity to sint
    get_raw_input_from = get_input_from
    @classmethod
    def load_dynamic_mem(cls, address):
        res = cls()
        if isinstance(address, int):
            inst.ldmsd(res, address, cls.n)
        else:
            inst.ldmsdi(res, address, cls.n)
        return res
    def store_in_dynamic_mem(self, address):
        if isinstance(address, int):
            inst.stmsd(self, address)
        else:
            inst.stmsdi(self, cbits.conv(address))
    def load_int(self, value):
        if (abs(value) > (1 << self.n)):
            raise Exception('public value %d longer than %d bits' \
                            % (value, self.n))
        if self.n <= 32:
            inst.ldbits(self, self.n, value)
        else:
            size = math.ceil(self.n / self.unit)
            tmp = regint(size=size)
            for i in range(size):
                tmp[i].load_int((value >> (i * 64)) % 2**64)
            self.load_other(tmp)
    def load_other(self, other):
        if isinstance(other, cbits) and self.n == other.n:
            inst.convcbit2s(self.n, self, other)
        else:
            super(sbits, self).load_other(other)
    @read_mem_value
    def __add__(self, other):
        if isinstance(other, int) or other is None:
            return self.xor_int(other)
        else:
            if not isinstance(other, sbits):
                other = self.conv(other)
            if self.n is None or other.n is None:
                assert self.n == other.n
                n = None
            else:
                n = min(self.n, other.n)
            res = self.new(n=n)
            inst.xors(n, res, self, other)
            if self.n != None and max(self.n, other.n) > n:
                if self.n > n:
                    longer = self
                else:
                    longer = other
                bits = res.bit_decompose() + longer.bit_decompose()[n:]
                res = self.bit_compose(bits)
            return res
    __radd__ = __add__
    __sub__ = __add__
    __xor__ = __add__
    __rxor__ = __add__
    @read_mem_value
    def __rsub__(self, other):
        if isinstance(other, cbits):
            return other + self
        else:
            return self.xor_int(other)
    @read_mem_value
    def __mul__(self, other):
        if isinstance(other, int):
            return self.mul_int(other)
        try:
            if min(self.n, other.n) != 1:
                raise NotImplementedError('high order multiplication')
            n = max(self.n, other.n)
            res = self.new(n=max(self.n, other.n))
            order = (self, other) if self.n != 1 else (other, self)
            inst.andrs(n, res, *order)
            return res
        except AttributeError:
            return NotImplemented
    __rmul__ = __mul__
    @read_mem_value
    def __and__(self, other):
        if util.is_zero(other):
            return 0
        elif util.is_all_ones(other, self.n) or \
             (other is None and self.n == None):
            return self
        res = self.new(n=self.n)
        if not isinstance(other, sbits):
            other = cbits.get_type(self.n).conv(other)
            inst.andm(self.n, res, self, other)
            return res
        other = self.conv(other)
        assert(self.n == other.n)
        inst.ands(self.n, res, self, other)
        return res
    __rand__ = __and__
    def xor_int(self, other):
        if other == 0:
            return self
        elif other == self.long_one():
            return ~self
        self_bits = self.bit_decompose()
        other_bits = util.bit_decompose(other, max(self.n, util.int_len(other)))
        extra_bits = [self.new(b, n=1) for b in other_bits[self.n:]]
        return self.bit_compose([~x if y else x \
                                 for x,y in zip(self_bits, other_bits)] \
                                + extra_bits)
    def mul_int(self, other):
        assert(util.is_constant(other))
        if other == 0:
            return 0
        elif other == 1:
            return self
        elif self.n == 1:
            bits = util.bit_decompose(other, util.int_len(other))
            zero = sbit(0)
            mul_bits = [self if b else zero for b in bits]
            return self.bit_compose(mul_bits)
        else:
            print(self.n, other)
            return NotImplemented
    def __lshift__(self, i):
        return self.bit_compose([sbit(0)] * i + self.bit_decompose()[:self.max_length-i])
    def __invert__(self):
        res = type(self)(n=self.n)
        inst.nots(self.n, res, self)
        return res
    def __neg__(self):
        return self
    def reveal(self):
        if self.n == None or \
           self.n > max(self.max_length, self.clear_type.max_length):
            assert(self.unit == self.clear_type.unit)
        res = self.clear_type.get_type(self.n)()
        inst.reveal(self.n, res, self)
        return res
    def equal(self, other, n=None):
        bits = (~(self + other)).bit_decompose()
        return reduce(operator.mul, bits)
    def right_shift(self, m, k, security=None, signed=True):
        return self.TruncPr(k, m)
    def TruncPr(self, k, m, kappa=None):
        if k > self.n:
            raise Exception('TruncPr overflow: %d > %d' % (k, self.n))
        bits = self.bit_decompose()
        res = self.get_type(k - m).bit_compose(bits[m:k])
        return res
    @classmethod
    def two_power(cls, n):
        if n > cls.n:
            raise Exception('two_power overflow: %d > %d' % (n, cls.n))
        res = cls()
        if n == cls.n:
            res.load_int(-1 << (n - 1))
        else:
            res.load_int(1 << n)
        return res
    def popcnt(self):
        """ Population count / Hamming weight.

        :return: :py:obj:`sbits` of required length """
        return sbitvec(self).popcnt().elements()[0]
    @classmethod
    def trans(cls, rows):
        rows = list(rows)
        if len(rows) == 1 and rows[0].n <= rows[0].unit:
            return rows[0].bit_decompose()
        n_columns = rows[0].n
        for row in rows:
            assert(row.n == n_columns)
        if n_columns == 1 and len(rows) <= cls.unit:
            return [cls.bit_compose(rows)]
        else:
            res = [cls.new(n=len(rows)) for i in range(n_columns)]
            inst.trans(len(res), *(res + rows))
            return res
    def if_else(self, x, y):
        """
        Vectorized oblivious selection::

            sb32 = sbits.get_type(32)
            print_ln('%s', sb32(3).if_else(sb32(5), sb32(2)).reveal())

        This will output 1.
        """
        return result_conv(x, y)(self & (x ^ y) ^ y)
    @staticmethod
    def bit_adder(*args, **kwargs):
        return sbitint.bit_adder(*args, **kwargs)
    @staticmethod
    def ripple_carry_adder(*args, **kwargs):
        return sbitint.ripple_carry_adder(*args, **kwargs)
    def to_sint(self, n_bits):
        """ Convert the :py:obj:`n_bits` least significant bits to
        :py:obj:`~Compiler.types.sint`. """
        bits = sbitvec.from_vec(sbitvec([self]).v[:n_bits]).elements()[0]
        bits = sint(bits, size=n_bits)
        return sint.bit_compose(bits)

class sbitvec(_vec):
    """ Vector of registers of secret bits, effectively a matrix of secret bits.
    This facilitates parallel arithmetic operations in binary circuits.
    Container types are not supported, use :py:obj:`sbitvec.get_type` for that.

    You can access the rows by member :py:obj:`v` and the columns by calling
    :py:obj:`elements`.

    There are three ways to create an instance:

    1. By transposition::

        sb32 = sbits.get_type(32)
        x = sbitvec([sb32(5), sb32(3), sb32(0)])
        print_ln('%s', [x.v[0].reveal(), x.v[1].reveal(), x.v[2].reveal()])
        print_ln('%s', [x.elements()[0].reveal(), x.elements()[1].reveal()])

       This should output::

        [3, 2, 1]
        [5, 3]

    2. Without transposition::

        sb32 = sbits.get_type(32)
        x = sbitvec.from_vec([sb32(5), sb32(3)])
        print_ln('%s', [x.v[0].reveal(), x.v[1].reveal()])

       This should output::

        [5, 3]

    3. From :py:obj:`~Compiler.types.sint`::

        y = sint(5)
        x = sbitvec(y, 3, 3)
        print_ln('%s', [x.v[0].reveal(), x.v[1].reveal(), x.v[2].reveal()])

       This should output::

        [1, 0, 1]
    """
    bit_extend = staticmethod(lambda v, n: v[:n] + [0] * (n - len(v)))
    @classmethod
    def get_type(cls, n):
        """ Create type for fixed-length vector of registers of secret bits.

        As with :py:obj:`sbitvec`, you can access the rows by member
        :py:obj:`v` and the columns by calling :py:obj:`elements`.
        """
        class sbitvecn(cls, _structure):
            @staticmethod
            def malloc(size, creator_tape=None):
                return sbit.malloc(size * n, creator_tape=creator_tape)
            @staticmethod
            def n_elements():
                return n
            @classmethod
            def get_input_from(cls, player):
                """ Secret input from :py:obj:`player`. The input is decomposed
                into bits.

                :param: player (int)
                """
                res = cls.from_vec(sbit() for i in range(n))
                inst.inputbvec(n + 3, 0, player, *res.v)
                return res
            get_raw_input_from = get_input_from
            @classmethod
            def from_vec(cls, vector):
                res = cls()
                res.v = _complement_two_extend(list(vector), n)[:n]
                return res
            def __init__(self, other=None, size=None):
                assert size in (None, 1)
                if other is not None:
                    if util.is_constant(other):
                        self.v = [sbit((other >> i) & 1) for i in range(n)]
                    elif isinstance(other, _vec):
                        self.v = self.bit_extend(other.v, n)
                    elif isinstance(other, (list, tuple)):
                        self.v = self.bit_extend(sbitvec(other).v, n)
                    else:
                        self.v = sbits(other, n=n).bit_decompose(n)
                    assert len(self.v) == n
            @classmethod
            def load_mem(cls, address):
                if not isinstance(address, int) and len(address) == n:
                    return cls.from_vec(sbit.load_mem(x) for x in address)
                else:
                    return cls.from_vec(sbit.load_mem(address + i)
                                        for i in range(n))
            def store_in_mem(self, address):
                for x in self.v:
                    assert util.is_constant(x) or x.n == 1
                v = [sbit.conv(x) for x in self.v]
                if not isinstance(address, int) and len(address) == n:
                    for x, y in zip(v, address):
                        x.store_in_mem(y)
                else:
                    for i in range(n):
                        v[i].store_in_mem(address + i)
            def reveal(self):
                revealed = [cbit() for i in range(len(self))]
                for i in range(len(self)):
                    try:
                        inst.reveal(1, revealed[i], self.v[i])
                    except:
                        revealed[i] = cbit.conv(self.v[i])
                return cbits.get_type(len(self)).bit_compose(revealed)
            @classmethod
            def two_power(cls, nn):
                return cls.from_vec([0] * nn + [1] + [0] * (n - nn - 1))
            def coerce(self, other):
                if util.is_constant(other):
                    return self.from_vec(util.bit_decompose(other, n))
                else:
                    return super(sbitvecn, self).coerce(other)
            @classmethod
            def bit_compose(cls, bits):
                bits = list(bits)
                if len(bits) < n:
                    bits += [0] * (n - len(bits))
                assert len(bits) == n
                return cls.from_vec(bits)
            def __str__(self):
                return 'sbitvec(%d)' % n
        return sbitvecn
    @classmethod
    def from_vec(cls, vector):
        res = cls()
        res.v = list(vector)
        return res
    compose = from_vec
    @classmethod
    def combine(cls, vectors):
        res = cls()
        res.v = sum((vec.v for vec in vectors), [])
        return res
    @classmethod
    def from_matrix(cls, matrix):
        # any number of rows, limited number of columns
        return cls.combine(cls(row) for row in matrix)
    def __init__(self, elements=None, length=None, input_length=None):
        if length:
            assert isinstance(elements, sint)
            if Program.prog.use_split():
                x = elements.split_to_two_summands(length)
                v = sbitint.carry_lookahead_adder(x[0], x[1], fewer_inv=True)
            else:
                prog = Program.prog
                if not prog.options.ring:
                    # force the use of edaBits
                    backup = prog.use_edabit()
                    prog.use_edabit(True)
                    from Compiler.floatingpoint import BitDecFieldRaw
                    self.v = BitDecFieldRaw(elements,
                                            input_length or prog.bit_length,
                                            length, prog.security)
                    prog.use_edabit(backup)
                    return
                l = int(Program.prog.options.ring)
                r, r_bits = sint.get_edabit(length, size=elements.size)
                c = ((elements - r) << (l - length)).reveal()
                c >>= l - length
                cb = [(c >> i) for i in range(length)]
                x = sbitintvec.from_vec(r_bits) + sbitintvec.from_vec(cb)
                v = x.v
            self.v = v[:length]
        elif elements is not None and not (util.is_constant(elements) and \
             elements == 0):
            self.v = sbits.trans(elements)
    def popcnt(self):
        """ Population count / Hamming weight.

        :return: :py:obj:`sbitintvec` of required length """
        res = sbitint.wallace_tree([[b] for b in self.v])
        while util.is_zero(res[-1]):
            del res[-1]
        return sbitintvec.get_type(len(res)).from_vec(res)
    def elements(self, start=None, stop=None):
        if stop is None:
            start, stop = stop, start
        return sbits.trans(self.v[start:stop])
    def coerce(self, other):
        if isinstance(other, cint):
            size = other.size
            return (other.get_vector(base, min(64, size - base)) \
                    for base in range(0, size, 64))
        return other
    def __xor__(self, other):
        other = self.coerce(other)
        return self.from_vec(x ^ y for x, y in zip(self.v, other))
    def __and__(self, other):
        return self.from_vec(x & y for x, y in zip(self.v, other.v))
    def if_else(self, x, y):
        assert(len(self.v) == 1)
        try:
            return self.from_vec(util.if_else(self.v[0], a, b) \
                                 for a, b in zip(x, y))
        except:
            return util.if_else(self.v[0], x, y)
    def __iter__(self):
        return iter(self.v)
    def __len__(self):
        return len(self.v)
    def __getitem__(self, index):
        return self.v[index]
    @classmethod
    def conv(cls, other):
        if isinstance(other, cls):
            return cls.from_vec(other.v)
        else:
            return cls(other)
    @property
    def size(self):
        if not self.v or util.is_constant(self.v[0]):
            return 1
        else:
            return self.v[0].n
    @property
    def n_bits(self):
        return len(self.v)
    def store_in_mem(self, address):
        for i, x in enumerate(self.elements()):
            x.store_in_mem(address + i)
    def bit_decompose(self, n_bits=None, security=None):
        return self.v[:n_bits]
    bit_compose = from_vec
    def reveal(self):
        assert len(self) == 1
        return self.v[0].reveal()
    def long_one(self):
        return [x.long_one() for x in self.v]
    def __rsub__(self, other):
        return self.from_vec(y - x for x, y in zip(self.v, other))
    def half_adder(self, other):
        other = self.coerce(other)
        res = zip(*(x.half_adder(y) for x, y in zip(self.v, other)))
        return (self.from_vec(x) for x in res)
    def __mul__(self, other):
        if isinstance(other, int):
            return self.from_vec(x * other for x in self.v)
    def __add__(self, other):
        return self.from_vec(x + y for x, y in zip(self.v, other))
    def bit_and(self, other):
        return self & other
    def bit_xor(self, other):
        return self ^ other
    def right_shift(self, m, k, security=None, signed=True):
        return self.from_vec(self.v[m:])

class bit(object):
    n = 1
    
def result_conv(x, y):
    try:
        if util.is_constant(x):
            if util.is_constant(y):
                return lambda x: x
            else:
                return type(y).conv
        if util.is_constant(y):
            return type(x).conv
        if type(x) is type(y):
            return type(x).conv
    except AttributeError:
        pass
    return lambda x: x

class sbit(bit, sbits):
    """ Single secret bit. """
    def if_else(self, x, y):
        """ Non-vectorized oblivious selection::

            sb32 = sbits.get_type(32)
            print_ln('%s', sbit(1).if_else(sb32(5), sb32(2)).reveal())

        This will output 5.
        """
        return result_conv(x, y)(self * (x ^ y) ^ y)

class cbit(bit, cbits):
    pass

sbits.bit_type = sbit
cbits.bit_type = cbit

class bitsBlock(oram.Block):
    value_type = sbits
    def __init__(self, value, start, lengths, entries_per_block):
        oram.Block.__init__(self, value, lengths)
        length = sum(self.lengths)
        used_bits = entries_per_block * length
        self.value_bits = self.value.bit_decompose(used_bits)
        start_length = util.log2(entries_per_block)
        self.start_bits = util.bit_decompose(start, start_length)
        self.start_demux = oram.demux_list(self.start_bits)
        self.entries = [sbits.bit_compose(self.value_bits[i*length:][:length]) \
                        for i in range(entries_per_block)]
        self.mul_entries = list(map(operator.mul, self.start_demux, self.entries))
        self.bits = sum(self.mul_entries).bit_decompose()
        self.mul_value = sbits.compose(self.mul_entries, sum(self.lengths))
        self.anti_value = self.mul_value + self.value
    def set_slice(self, value):
        value = sbits.compose(util.tuplify(value), sum(self.lengths))
        for i,b in enumerate(self.start_bits):
            value = b.if_else(value << (2**i * sum(self.lengths)), value)
        self.value = value + self.anti_value
        return self

oram.block_types[sbits] = bitsBlock

class dyn_sbits(sbits):
    pass

class DynamicArray(Array):
    def __init__(self, *args):
        Array.__init__(self, *args)
    def _malloc(self):
        return Program.prog.malloc(self.length, 'sd', self.value_type)
    def _load(self, address):
        return self.value_type.load_dynamic_mem(cbits.conv(address))
    def _store(self, value, address):
        if isinstance(value, MemValue):
            value = value.read()
        if isinstance(value, sbits):
            self.value_type.conv(value).store_in_dynamic_mem(address)
        else:
            cbits.conv(value).store_in_dynamic_mem(address)

sbits.dynamic_array = DynamicArray
cbits.dynamic_array = Array

def _complement_two_extend(bits, k):
    return bits + [bits[-1]] * (k - len(bits))

class _sbitintbase:
    def extend(self, n):
        bits = self.bit_decompose()
        bits += [bits[-1]] * (n - len(bits))
        return self.get_type(n).bit_compose(bits)
    def cast(self, n):
        bits = self.bit_decompose()[:n]
        bits += [bits[-1]] * (n - len(bits))
        return self.get_type(n).bit_compose(bits)
    def round(self, k, m, kappa=None, nearest=None, signed=None):
        bits = self.bit_decompose()
        if signed:
            bits += [bits[-1]] * (k - len(bits))
        res_bits = self.bit_adder(bits[m:k], [bits[m-1]])
        return self.get_type(k - m).compose(res_bits)
    def int_div(self, other, bit_length=None):
        k = bit_length or max(self.n, other.n)
        return (library.IntDiv(self.extend(k), other.extend(k), k) >> k).cast(k)
    def Norm(self, k, f, kappa=None, simplex_flag=False):
        absolute_val = abs(self)
        #next 2 lines actually compute the SufOR for little indian encoding
        bits = absolute_val.bit_decompose(k)[::-1]
        suffixes = floatingpoint.PreOR(bits)[::-1]
        z = [0] * k
        for i in range(k - 1):
            z[i] = suffixes[i] - suffixes[i+1]
        z[k - 1] = suffixes[k-1]
        z.reverse()
        t2k = self.get_type(2 * k)
        acc = t2k.bit_compose(z)
        sign = self.bit_decompose()[-1]
        signed_acc = util.if_else(sign, -acc, acc)
        absolute_val_2k = t2k.bit_compose(absolute_val.bit_decompose())
        part_reciprocal = absolute_val_2k * acc
        return part_reciprocal, signed_acc
    def pow2(self, k):
        l = int(math.ceil(math.log(k, 2)))
        bits = [self.equal(i, l) for i in range(k)]
        return self.get_type(k).bit_compose(bits)

class sbitint(_bitint, _number, sbits, _sbitintbase):
    """ Secret signed integer in one binary register. Use :py:obj:`get_type()`
    to specify the bit length::

        si32 = sbitint.get_type(32)
        print_ln('add: %s', (si32(5) + si32(3)).reveal())
        print_ln('sub: %s', (si32(5) - si32(3)).reveal())
        print_ln('mul: %s', (si32(5) * si32(3)).reveal())
        print_ln('lt: %s', (si32(5) < si32(3)).reveal())

    This should output::

        add: 8
        sub: 2
        mul: 15
        lt: 0

    """
    n_bits = None
    bin_type = None
    types = {}
    vector_mul = True
    @classmethod
    def get_type(cls, n, other=None):
        """ Returns a signed integer type with fixed length.

        :param n: length """
        if isinstance(other, sbitvec):
            return sbitvec
        if n in cls.types:
            return cls.types[n]
        sbits_type = sbits.get_type(n)
        class _(sbitint, sbits_type):
            # n_bits is used by _bitint
            n_bits = n
            bin_type = sbits_type
        _.__name__ = 'sbitint' + str(n)
        cls.types[n] = _
        return _
    @classmethod
    def combo_type(cls, other):
        if isinstance(other, sbitintvec):
            return sbitintvec
        else:
            return cls
    @classmethod
    def new(cls, value=None, n=None):
        return cls.get_type(n)(value)
    def set_length(*args):
        pass
    @classmethod
    def bit_compose(cls, bits):
        # truncate and extend bits
        bits = list(bits)[:cls.n]
        bits += [0] * (cls.n - len(bits))
        return super(sbitint, cls).bit_compose(bits)
    def force_bit_decompose(self, n_bits=None):
        return sbits.bit_decompose(self, n_bits)
    def TruncMul(self, other, k, m, kappa=None, nearest=False):
        if nearest:
            raise CompilerError('round to nearest not implemented')
        self_bits = self.bit_decompose()
        other_bits = other.bit_decompose()
        if len(self_bits) + len(other_bits) > k:
            raise Exception('invalid parameters for TruncMul: '
                            'self:%d, other:%d, k:%d' %
                            (len(self_bits), len(other_bits), k))
        t = self.get_type(k)
        a = t.bit_compose(self_bits + [self_bits[-1]] * (k - len(self_bits)))
        t = other.get_type(k)
        b = t.bit_compose(other_bits + [other_bits[-1]] * (k - len(other_bits)))
        product = a * b
        res_bits = product.bit_decompose()[m:k]
        res_bits += [res_bits[-1]] * (self.n - len(res_bits))
        t = self.combo_type(other).get_type(k - m)
        return t.bit_compose(res_bits)
    def __mul__(self, other):
        if isinstance(other, sbitintvec):
            return other * self
        else:
            return super(sbitint, self).__mul__(other)
    @classmethod
    def get_bit_matrix(cls, self_bits, other):
        n = len(self_bits)
        assert n == other.n
        res = []
        for i, bit in enumerate(self_bits):
            if util.is_zero(bit):
                res.append([0] * (n - i))
            else:
                if cls.vector_mul:
                    x = sbits.get_type(n - i)()
                    inst.andrs(n - i, x, other, bit)
                    res.append(x.bit_decompose(n - i))
                else:
                    res.append([(x & bit) for x in other.bit_decompose(n - i)])
        return res
    @classmethod
    def popcnt_bits(cls, bits):
        res = sbitintvec.popcnt_bits(bits).elements()[0]
        res = cls.conv(res)
        return res
    def pow2(self, k):
        """ Computer integer power of two.

        :param k: bit length of input """
        return _sbitintbase.pow2(self, k)

class sbitintvec(sbitvec, _number, _bitint, _sbitintbase):
    """
    Vector of signed integers for parallel binary computation::

        sb32 = sbits.get_type(32)
        siv32 = sbitintvec.get_type(32)
        a = siv32([sb32(3), sb32(5)])
        b = siv32([sb32(4), sb32(6)])
        c = (a + b).elements()
        print_ln('add: %s, %s', c[0].reveal(), c[1].reveal())
        c = (a * b).elements()
        print_ln('mul: %s, %s', c[0].reveal(), c[1].reveal())
        c = (a - b).elements()
        print_ln('sub: %s, %s', c[0].reveal(), c[1].reveal())
        c = (a < b).bit_decompose()
        print_ln('lt: %s, %s', c[0].reveal(), c[1].reveal())

    This should output::

        add: 7, 11
        mul: 12, 30
        sub: -1, 11
        lt: 1, 1

    """
    bit_extend = staticmethod(_complement_two_extend)
    @classmethod
    def popcnt_bits(cls, bits):
        return sbitvec.from_vec(bits).popcnt()
    def elements(self):
        return [sbitint.get_type(len(self.v))(x)
                for x in sbitvec.elements(self)]
    def __add__(self, other):
        if util.is_zero(other):
            return self
        other = self.coerce(other)
        assert(len(self.v) == len(other.v))
        v = sbitint.bit_adder(self.v, other.v)
        return self.from_vec(v)
    __radd__ = __add__
    def __mul__(self, other):
        if isinstance(other, sbits):
            return self.from_vec(other * x for x in self.v)
        elif isinstance(other, sbitfixvec):
            return NotImplemented
        matrix = []
        for i, b in enumerate(util.bit_decompose(other)):
            matrix.append([x & b for x in self.v[:len(self.v)-i]])
        v = sbitint.wallace_tree_from_matrix(matrix)
        return self.from_vec(v[:len(self.v)])
    __rmul__ = __mul__
    reduce_after_mul = lambda x: x
    def TruncMul(self, other, k, m, kappa=None, nearest=False):
        if nearest:
            raise CompilerError('round to nearest not implemented')
        if not isinstance(other, sbitintvec):
            other = sbitintvec(other)
        assert len(self.v) + len(other.v) <= k
        a = self.get_type(k).from_vec(_complement_two_extend(self.v, k))
        b = self.get_type(k).from_vec(_complement_two_extend(other.v, k))
        tmp = a * b
        assert len(tmp.v) == k
        return self.get_type(k - m).from_vec(tmp[m:])
    def pow2(self, k):
        """ Computer integer power of two.

        :param k: bit length of input """
        return _sbitintbase.pow2(self, k)

sbitint.vec = sbitintvec

class cbitfix(object):
    malloc = staticmethod(lambda *args: cbits.malloc(*args))
    n_elements = staticmethod(lambda: 1)
    conv = staticmethod(lambda x: x)
    load_mem = classmethod(lambda cls, *args: cls._new(cbits.load_mem(*args)))
    store_in_mem = lambda self, *args: self.v.store_in_mem(*args)
    @classmethod
    def _new(cls, value):
        res = cls()
        res.v = value
        return res
    def output(self):
        v = self.v
        if self.k < v.unit:
            bits = self.v.bit_decompose(self.k)
            sign = bits[-1]
            v += (sign << (self.k)) * -1
        inst.print_float_plainb(v, cbits(-self.f, n=32), cbits(0), cbits(0), cbits(0))

class sbitfix(_fix):
    """ Secret signed integer in one binary register.
    Use :py:obj:`set_precision()` to change the precision.

    Example::

        print_ln('add: %s', (sbitfix(0.5) + sbitfix(0.3)).reveal())
        print_ln('mul: %s', (sbitfix(0.5) * sbitfix(0.3)).reveal())
        print_ln('sub: %s', (sbitfix(0.5) - sbitfix(0.3)).reveal())
        print_ln('lt: %s', (sbitfix(0.5) < sbitfix(0.3)).reveal())

    will output roughly::

        add: 0.800003
        mul: 0.149994
        sub: 0.199997
        lt: 0

    """
    float_type = type(None)
    clear_type = cbitfix
    @classmethod
    def set_precision(cls, f, k=None):
        super(sbitfix, cls).set_precision(f, k)
        cls.int_type = sbitint.get_type(cls.k)
    @classmethod
    def load_mem(cls, address, size=None):
        if size not in (None, 1):
            v = [cls.int_type.load_mem(address + i) for i in range(size)]
            return sbitfixvec._new(sbitintvec(v))
        else:
            return super(sbitfix, cls).load_mem(address)
    @classmethod
    def get_input_from(cls, player):
        """ Secret input from :py:obj:`player`.

        :param: player (int)
        """
        v = cls.int_type()
        inst.inputb(player, cls.k, cls.f, v)
        return cls._new(v)
    def __xor__(self, other):
        return type(self)._new(self.v ^ other.v)
    def __mul__(self, other):
        if isinstance(other, sbit):
            return type(self)._new(self.int_type(other * self.v))
        elif isinstance(other, sbitfixvec):
            return other * self
        else:
            return super(sbitfix, self).__mul__(other)
    __rxor__ = __xor__
    __rmul__ = __mul__
    @staticmethod
    def multipliable(other, k, f, size):
        class cls(_fix):
            int_type = sbitint.get_type(k)
            clear_type = cbitfix
        cls.set_precision(f, k)
        return cls._new(cls.int_type(other), k, f)

class sbitfixvec(_fix):
    """ Vector of fixed-point numbers for parallel binary computation.

    Use :py:obj:`set_precision()` to change the precision.

    Example::

        a = sbitfixvec([sbitfix(0.3), sbitfix(0.5)])
        b = sbitfixvec([sbitfix(0.4), sbitfix(0.6)])
        c = (a + b).elements()
        print_ln('add: %s, %s', c[0].reveal(), c[1].reveal())
        c = (a * b).elements()
        print_ln('mul: %s, %s', c[0].reveal(), c[1].reveal())
        c = (a - b).elements()
        print_ln('sub: %s, %s', c[0].reveal(), c[1].reveal())
        c = (a < b).bit_decompose()
        print_ln('lt: %s, %s', c[0].reveal(), c[1].reveal())

    This should output roughly::

        add: 0.699997, 1.10001
        mul: 0.119995, 0.300003
        sub: -0.0999908, -0.100021
        lt: 1, 1

    """
    int_type = sbitintvec.get_type(sbitfix.k)
    float_type = type(None)
    clear_type = cbitfix
    @classmethod
    def set_precision(cls, f, k=None):
        super(sbitfixvec, cls).set_precision(f=f, k=k)
        cls.int_type = sbitintvec.get_type(cls.k)
    @classmethod
    def get_input_from(cls, player):
        """ Secret input from :py:obj:`player`.

        :param: player (int)
        """
        v = [sbit() for i in range(sbitfix.k)]
        inst.inputbvec(len(v) + 3, sbitfix.f, player, *v)
        return cls._new(cls.int_type.from_vec(v))
    def __init__(self, value=None, *args, **kwargs):
        if isinstance(value, (list, tuple)):
            self.v = self.int_type.from_vec(sbitvec([x.v for x in value]))
        else:
            super(sbitfixvec, self).__init__(value, *args, **kwargs)
    def elements(self):
        return [sbitfix._new(x, f=self.f, k=self.k) for x in self.v.elements()]
    def mul(self, other):
        if isinstance(other, sbits):
            return self._new(self.v * other)
        else:
            return super(sbitfixvec, self).mul(other)
    def __xor__(self, other):
        return self._new(self.v ^ other.v)
    @staticmethod
    def multipliable(other, k, f, size):
        class cls(_fix):
            int_type = sbitint.get_type(k)
            clear_type = cbitfix
        cls.set_precision(f, k)
        return cls._new(cls.int_type(other), k, f)

sbitfix.set_precision(16, 31)
sbitfixvec.set_precision(16, 31)

sbitfix.vec = sbitfixvec

class cbitfloat:
    def __init__(self, v, p, z, s, nan=0):
        self.v, self.p, self.z, self.s, self.nan = v, p, z, s, cbit.conv(nan)

    def output(self):
        inst.print_float_plainb(self.v, self.p, self.z, self.s, self.nan)
