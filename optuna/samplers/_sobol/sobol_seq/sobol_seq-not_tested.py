"""
  Licensing:
    This code is distributed under the MIT license.

  Authors:
    Original FORTRAN77 version of tau_sobol by Bennett Fox.
    MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari

    Original MATLAB versions of other functions by John Burkardt.
    PYTHON versions by Corrado Chisari

    Original code is available from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
"""

import numpy as np


def r4_uniform_01(seed):
    """
    r4_uniform_01 is a uniform random number generator.

    Discussion:
      This routine implements the recursion

        seed = 16807 * seed mod ( 2**31 - 1 )
        r4_uniform_01 = seed / ( 2**31 - 1 )

      The integer arithmetic never requires more than 32 bits,
      including a sign bit.

      If the initial seed is 12345, then the first three computations are

       +------------+-------------+--------------
       | Input SEED | Output SEED | R4_UNIFORM_01
       +------------+-------------+--------------
       |      12345 |   207482415 |      0.096616
       |  207482415 |  1790989824 |      0.833995
       | 1790989824 |  2035175616 |      0.947702

    Reference:
      Paul Bratley, Bennett Fox, Linus Schrage,
      A Guide to Simulation,
      Springer Verlag, pages 201-202, 1983.

      Pierre L'Ecuyer,
      Random Number Generation,
      in Handbook of Simulation,
      edited by Jerry Banks,
      Wiley Interscience, page 95, 1998.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Peter Lewis, Allen Goodman, James Miller,
      A Pseudo-Random Number Generator for the System/360,
      IBM Systems Journal,
      Volume 8, pages 136-143, 1969.

    Parameters:
      Input, integer SEED, the integer "seed" used to generate
      the output random number.  SEED should not be 0.

      Output, real R, a random value between 0 and 1.
      Output, integer SEED, the updated seed.  This would
      normally be used as the input seed on the next call.
    """

    seed = np.floor(seed)

    seed = np.mod(seed, 2147483647)

    if seed < 0:
        seed = seed + 2147483647

    k = np.floor(seed / 127773)

    seed = 16807 * (seed - k * 127773) - k * 2836

    if seed < 0:
        seed = seed + 2147483647

    r = seed * 4.656612875e-10

    return [r, seed]


def r8mat_write(output_filename, m, n, table):
    """
    r8mat_write writes an R8MAT file.

    Discussion:
      An R8MAT is an array of R8's.

    Parameters:
      Input, string OUTPUT_FILENAME, the output filename.
      Input, integer M, the spatial dimension.
      Input, integer N, the number of points.
      Input, real TABLE(M,N), the points.
    """
    # Open the file.
    try:
        output_unit = open(output_filename, "w")
    except:
        print("R8MAT_WRITE - Error!")
        print("  Could not open the output file.")
        return

    #  Write the data.
    #  For smaller data files, and less precision, try:
    # fprintf ( output_unit, '  #14.6f', table(i,j) )
    for j in range(0, n):
        for i in range(0, m):
            output_unit.write("  %24.16f" % table[i, j])
        output_unit.write("\n")

    #  Close the file.
    output_unit.close()

    return


def tau_sobol(dim_num):
    """
    tau_sobol defines favorable starting seeds for Sobol sequences.

    Discussion:
      For spatial dimensions 1 through 13, this routine returns
      a "favorable" value TAU by which an appropriate starting point
      in the Sobol sequence can be determined.

      These starting points have the form N = 2**K, where
      for integration problems, it is desirable that
        TAU + DIM_NUM - 1 <= K
      while for optimization problems, it is desirable that
        TAU < K.

    Reference:
      IA Antonov, VM Saleev,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 19, 1980, pages 252 - 256.

      Paul Bratley, Bennett Fox,
      Algorithm 659:
      Implementing Sobol's Quasirandom Sequence Generator,
      ACM Transactions on Mathematical Software,
      Volume 14, Number 1, pages 88-100, 1988.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Stephen Joe, Frances Kuo
      Remark on Algorithm 659:
      Implementing Sobol's Quasirandom Sequence Generator,
      ACM Transactions on Mathematical Software,
      Volume 29, Number 1, pages 49-57, March 2003.

      Ilya Sobol,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 16, pages 236-242, 1977.

      Ilya Sobol, YL Levitan,
      The Production of Points Uniformly Distributed in a Multidimensional
      Cube (in Russian),
      Preprint IPM Akad. Nauk SSSR,
      Number 40, Moscow 1976.

    Parameters:
      Input, integer DIM_NUM, the spatial dimension.  Only values
      of 1 through 13 will result in useful responses.

      Output, integer TAU, the value TAU.
    """
    dim_max = 13

    tau_table = [0, 0, 1, 3, 5, 8, 11, 15, 19, 23, 27, 31, 35]

    if 1 <= dim_num and dim_num <= dim_max:
        tau = tau_table[dim_num]
    else:
        tau = -1

    return tau
