# AccelInterfaces

[![Build Status](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml?query=branch%3Amain)

**AccelInterfaces** is a GPU and CPU programming interfaces for [Julia](http://julialang.org/) programmers.

**AccelInterfaces**, or JAI(Julia Accelerator Interfaces) focus on reusing legacy Fortran and C/C++(under development) applications which may already ported to GPU using compiler-based frameworks such as Cuda, Hip, OpenAcc, OpenMP target, OpenCL, and so on.

This package is still in early phase of development. Only a subset of mentioned features are developed. Please use this at your own risk.

## Package features

- Generates ccall arguments to shared library
- Creates shared library from user's Fortran/C/C++ code
- User interface is simplied by using macros

## Installation

```julia
Pkg.add("AccelInterfaces")
```

## Quickstart

To create a plot it's as simple as:

```julia
using AccelInterfaces
kernel_text = """
[fortran]

INTEGER i

DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO
"""

const N = 3
const x = fill(1, N)
const y = fill(2, N)
const z = fill(0, N)
const answer = fill(3, N)

@jaccel myaccel framework(fortran) compile("gfortran -fPIC -shared") 

@jkernel mykernel myaccel kernel_text

@jlaunch(mykernel, x, y; output=(z,))

@test z == answer
```

## Documentation

T.B.D.
