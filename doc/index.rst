欢迎阅读MP-SPDZ中文文档!
===================================

本文档展示了MP-SPDZ编译器提供的最重要的高级功能参考. 有关如何运行程序,
已实现协议的教程和文档等事项, 请参见原文https://github.com/data61/MP-SPDZ.

Compilation process
-------------------

After putting your code in ``Program/Source/<progname>.mpc``, run the
compiler from the root directory as follows

.. code-block:: bash

  ./compile.py [options] <progname> [args]

The arguments ``<progname> [args]`` are accessible as list under
``program.args`` within ``progname.mpc``, with ``<progname>`` as
``program.args[0]``.

The following options influence the computation domain:

.. cmdoption:: -F <integer length>
	       --field=<integer length>

   Compile for computation modulo a prime and the default integer
   length. This means that secret integers are assumed to have at most
   said length unless explicitly told otherwise. The compiled output
   will communicate the minimum length of the prime number to the
   virtual machine, which will fail if this is not met. This is the
   default with an *integer length* set to 64. When not specifying the
   prime, the minimum prime length will be around 40 bits longer than
   the integer length. Furthermore, the computation will be optimistic
   in the sense that overflows in the secrets might have security
   implications.

.. cmdoption:: -P <prime>
	       --prime=<prime>

   Specify a concrete prime modulus for computation. This can be used
   together with :option:`-F`, in which case *integer length* has to
   be at most the prime length minus two. The security implications of
   overflows in the secrets do not go beyond incorrect results.

.. cmdoption:: -R <ring size>
	       --ring=<ring size>

   Compile for computation modulo 2^(*ring size*). This will set the
   assumed length of secret integers to one less because many
   operations require this. The exact ring size will be communicated
   to the virtual machine, which will use it automatically if
   supported.

.. cmdoption:: -B <integer length>
	       --binary=<integer length>

   Compile for binary computation using *integer length* as default.

For arithmetic computation (:option:`-F`, :option:`-P`, and
:option:`-R`) you can set the bit
length during execution using ``program.set_bit_length(length)``. For
binary computation you can do so with ``sint =
sbitint.get_type(length)``.
Use :py:func:`sfix.set_precision` to change the range for fixed-point
numbers.

The following options switch from a single computation domain to
mixed computation when using in conjunction with arithmetic
computation:

.. cmdoption:: -X
	       --mixed

   Enables mixed computation using daBits.

.. cmdoption:: -Y
	       --edabit

   Enables mixed computation using edaBits.

The implementation of both daBits and edaBits are explained in this paper_.

.. _paper: https://eprint.iacr.org/2020/338

.. cmdoption:: -Z <number of parties>
	       --split=<number of parties>

   Enables mixed computation using local conversion. This has been
   used by `Mohassel and Rindal <https://eprint.iacr.org/2018/403>`_
   and `Araki et al. <https://eprint.iacr.org/2018/762>`_ It only
   works with additive secret sharing modulo a power of two.

The following options change less fundamental aspects of the
computation:

.. cmdoption:: -D
	       --dead-code-elimination

   Eliminates unused code. This currently means computation that isn't
   used for input or output or written to the so-called memory (e.g.,
   :py:class:`~Compiler.types.Array`; see :py:mod:`~Compiler.types`).

.. cmdoption:: -b <budget>
	       --budget=<budget>

   Set the budget for loop unrolling with
   :py:func:`~Compiler.library.for_range_opt` and similar. This means
   that loops are unrolled up to *budget* instructions. Default is
   100,000 instructions.

.. cmdoption:: -C
	       --CISC

   Speed up the compilation of repetitive code at the expense of a
   potentially higher number of communication rounds. For example, the
   compiler by default will try to compute a division and a logarithm
   in parallel if possible. Using this option complex operations such
   as these will be separated and only multiple divisions or
   logarithms will be computed in parallel. This speeds up the
   compilation because of reduced complexity.

Compilation vs run time
~~~~~~~~~~~~~~~~~~~~~~~

The most important thing to keep in mind is that the Python code is
executed at compile-time. This means that Python data structures such
as :py:class:`list` and :py:class:`dict` only exist at compile-time
and that all Python loops are unrolled. For run-time loops and lists,
you can use :py:func:`~Compiler.library.for_range` (or the more
optimizing :py:func:`~Compiler.library.for_range_opt`) and
:py:class:`~Compiler.types.Array`. For convenient multithreading you
can use :py:func:`~Compiler.library.for_range_opt_multithread`, which
automatically distributes the computation on the requested number of
threads.

This reference uses the term 'compile-time' to indicate Python types
(which are inherently known when compiling). If the term 'public' is
used, this means both compile-time values as well as public run-time
types such as :py:class:`~Compiler.types.regint`.

Reference
---------

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   Compiler
   instructions
   low-level


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
