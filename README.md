# AccelInterfaces

[![Build Status](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml?query=branch%3Amain)

**AccelInterfaces** is a GPU and CPU programming interface for [Julia](http://julialang.org/) programmers.

**AccelInterfaces**, or JAI(Julia Accelerator Interfaces), focuses on reusing legacy Fortran and C/C++(under development) applications. The legacy application may include directive based GPU programming such as OpenAcc or OpenMP target.

This package is still in the early phase of development. Only a subset of mentioned features are developed. Please use this package at your own risk.

## Package features

- Creates a shared library from pre-existing Fortran/C/C++ code(C/C++ is not supported yet)
- Generates arguments for [ccall](https://docs.julialang.org/en/v1/base/c/#ccall) function that uses the created shared library
- User interface is simplified by using Julia macros

## Installation

```julia
Pkg.add("AccelInterfaces")
```

## Quickstart

The following Julia code calculates a vector sum whose main algorithm is written in Fortran.

```julia
using AccelInterfaces

kernel_text = """

[fortran]

INTEGER i

DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO

[fortran_openacc]

INTEGER i

!\$acc parallel loop present(x, y, z)
DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO
!\$acc end parallel loop

"""

const N = 10
const x = fill(1, N)
const y = fill(2, N)
const z = fill(0, N)
const answer = fill(3, N)

@jaccel myaccel framework(fortran) compile("gfortran -fPIC -shared") 

@jkernel mykernel1 myaccel kernel_text

@jlaunch(mykernel1, x, y; output=(z,))

@assert z == answer
```

"kernel_text" variable contains a Fortran DO loop that actually calculates the vector sum. There are two versions of DO loop: Fortran and Fortran_OpenAcc. Users can select one of them using the "framework" clause of "@jaccel" Jai directive explained below.

"@jaccel" creates a Jai acceleration context. To identify the context, here we use the literal name of "myaccel". "framework" clause specifies the kind of acceleration(fortran in this example). With a compile clause, the user can provide Jai with the actual compiler command line to generate a shared library. The command line should include the compiler and all compiler flags except the "-o" flag with the name of an output file and the path to an input source file.

"@jkernel" creates a Jai kernel context. To identify the kernel context, here we uses the literal name of "mykernel". The last clause is the kernel program written in Fortran or C/C++. User can provide Jai with the kernel program in Julia string or external file path.

"@jlaunch" uses syntax similar to function calls with a pair of parentheses. Note that there should not be a space between "@jlaunch" and "(mykernel...". The first argument is the name of kernel context. All the variable names right before the semicolon are input variables to the kernel. "output" keyword argument specifies the names of output variables in a Julia Tuple.

Please note that you should use only simple variable names for inputs and outputs to/from the kernel in "@jlaunch". For example, you can not write like this: "@jlaunch(mykernel, x+1, func(y); output=(z::Vector,))."


To use GPU, you need to add additional Jai directives such as "@jenterdata", "@jexitdata", and "@jdecel". 

```julia
fill!(z, 0)

@jkernel mykernel2 myaccel kernel_text

@jenterdata myaccel allocate(x, y, z) update(x, y)

@jlaunch(mykernel2, x, y; output=(z,))

@jexitdata myaccel update(z) deallocate(x, y, z)

@jdecel myaccel

@assert z == answer
```
First of all, please note that the sequence of "@j* directives" is similar to the above example when we use the Fortran framework. Here we use "fortran_openacc" for "framework" clause which let Jai choose the content under "[fortran_openacc]" instead of "[fortran]" of kernel_text text.

The first clause to "@jenterdata" is the literal name defined in "@jaccel" in the above Fortran example. "allocate" clause allocates device memory for the variables of "x", "y", and "z". "update" clause copies the content of "x" and "y" to the allocated corresponding device variables.

In "@jexitdata", users can copy back data from the device using "update" clause. "deallocate" clause deallocates device memory allocated for "x", "y", and "z".

"@jdecel" directive notifies Jai that the user will not use "myaccel" context anymore.
