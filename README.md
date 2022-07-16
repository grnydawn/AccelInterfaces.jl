# AccelInterfaces

[![Build Status](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml?query=branch%3Amain)

**AccelInterfaces** is a GPU and CPU programming interfaces for [Julia](http://julialang.org/) programmers.

**AccelInterfaces**, or JAI(Julia Accelerator Interfaces) focus on reusing legacy Fortran and C/C++(under development) applications which may already ported to GPU using compiler-based frameworks such as Cuda, Hip, OpenAcc, OpenMP target, OpenCL, and so on.

This package is still in early phase of development. Only a subset of mentioned features are developed. Please use this at your own risk.

## Package features

- Creates a shared library from user's Fortran/C/C++ code
- Generates ccall arguments to use the crated shared library
- User interface is simplied by using Julia macros

## Installation

```julia
Pkg.add("AccelInterfaces")
```

## Quickstart

The following Julia code calculates a vector sum that is written in Fortran.

```julia
using AccelInterfaces

kernel_text = """
[fortran]

INTEGER i

DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO
"""

const N = 10
const x = fill(1, N)
const y = fill(2, N)
const z = fill(0, N)
const answer = fill(3, N)

@jaccel myaccel framework(fortran) compile("gfortran -fPIC -shared") 

@jkernel mykernel myaccel kernel_text

@jlaunch(mykernel, x, y; output=(z,))

@assert z == answer
```

"kernel_text" String variable contains a Fortran DO loop that actually calculates the vector sum.

"@jaccel" creates a Jai acceleration context. To identify the context, here we uses "myaccel". framework clause specifies the kind of acceleration(fortran in this example). With compile clause, user can provides the actual command line for generating a shared library. The command line should include all compiler flags except the name of output file and the name of input source file.

"@jkernel" creates a Jai kernel context. To identify the kernel context, here we uses "mykernel". The last clause is the external program written in Fortran or C/C++. Uesr can provide Jai with the external program in Julia String or external file path.

"@jlaunch" uses syntax similar to function call with a pair of parentheses. Note that there should not be a space between "@jlaunch" and "(mykernel...". The first argument is the name of kernel context. All the variable names righ before the semicolon are input variables to the kernel. "output" keyword argument should be a Julia Tuple.

Please note that you can not use Julia expression other than the names of variables when using "@jlaunch". For example, you can not write like this: "@jlaunch(mykernel, x+1, func(y); output=(z::Vector,))."


To use GPU, you need to add additional Jai directives such as "@jenterdata", "@jexitdata", and "@jdecel". 

```julia
using AccelInterfaces

kernel_text = """
[fortran]

INTEGER i

!$acc parallel loop present(x, y, z)
DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO
!$acc end parallel loop
"""

const N = 10
const x = fill(1, N)
const y = fill(2, N)
const z = fill(0, N)
const answer = fill(3, N)

@jkernel mykernel myaccel kernel_text

@jenterdata myaccel allocate(x, y, z) update(x, y)

@jlaunch(mykernel, x, y; output=(z,))

@jexitdata myaccel update(z) deallocate(x, y, z)

@jdecel myaccel

@assert z == answer
```

## Documentation

T.B.D.
