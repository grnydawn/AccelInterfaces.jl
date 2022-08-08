# Jai: Julia Accelerator Interfaces

[![Build Status](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Jai** is a GPU and CPU programming interface for [Julia](http://julialang.org/) programmers.

**Jai** focuses on reusing Fortran and C/C++ application codes. The codes may include directive based GPU programming such as OpenAcc and OpenMP Target.

This package is still in the early phase of development. Only a subset of mentioned features are developed. Please use this package at your own risk.

## Package features

- Creates a shared library from pre-existing Fortran/C/C++ code
- Generates arguments for [ccall](https://docs.julialang.org/en/v1/base/c/#ccall) function that uses the created shared library
- Simplifies User interface using Julia macros
- Takes advantages of Just-in-time(JIT) compilations

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

@jaccel myaccel1 framework(fortran) compile("gfortran -fPIC -shared")

@jkernel mykernel1 myaccel1 kernel_text

@jlaunch(mykernel1, x, y; output=(z,))

@assert z == answer
```

"kernel_text" variable contains a Fortran DO loop that actually calculates the vector sum. There are two versions of DO loop: Fortran and Fortran_OpenAcc. Users can select one of them using the "framework" clause of "@jaccel" Jai directive explained below.

"@jaccel" creates a Jai accelerator context. To identify the context, here we use the literal name of "myaccel1". "framework" clause specifies the kind of acceleration(fortran in this example). With a compile clause, the user can provide Jai with the actual compiler command line to generate a shared library. The command line should include the compiler and all compiler flags except the "-o" flag with the name of an output file and the path to an input source file.

"@jkernel" creates a Jai kernel context. To identify the kernel context, here we uses the literal name of "mykernel1". The last clause is the kernel program written in Fortran. User can provide Jai with the kernel program in Julia string or external file path.

"@jlaunch" uses syntax similar to function calls with a pair of parentheses. Note that there should not be a space between "@jlaunch" and "(mykernel1...". The first argument is the name of kernel context. All the variable names right before the semicolon are input variables to the kernel. "output" keyword argument specifies the names of output variables in a Julia Tuple.

Please note that you should use only simple variable names for inputs and outputs to/from the kernel in "@jlaunch". For example, you can not write like this: "@jlaunch(mykernel1, x+1, func(y); output=(z::Vector,))."


To use GPU, you need to add additional Jai directives such as "@jenterdata", "@jexitdata", and "@jdecel". 

```julia
fill!(z, 0)

@jaccel framework(fortran_openacc) compile("ftn -h acc,noomp -fPIC -shared")

@jkernel mykernel2 kernel_text

@jenterdata allocate(x, y, z) updateto(x, y)

@jlaunch(mykernel2, x, y; output=(z,))

@jexitdata updatefrom(z) deallocate(x, y, z)

@jdecel

@assert z == answer
```

Similar to above Fortran example, we use "@jaccel" directive to create Jai accelerator context. In this example, we used Cray compiler wrapper to compile Fortran program with OpenAcc. But you may modify the compile command for your needs. we use "fortran_openacc" for "framework" clause which let Jai choose the content under "[fortran_openacc]" instead of "[fortran]" of kernel_text text. Please note that we did not add the literal name for Jai accelerator context. Without specifying the name for Jai accelerator context, Jai creates a default Jai accelerator name (jai_accel_default) for you. you can skip specifying the default name in the following Jai directives as shown in this example.

"@jkernel" directive creates a Jai kernel context with the literal name of "mykernel2."

"allocate" clause in "@jenterdata" allocates device memory for the variables of "x", "y", and "z". "updateto" clause copies the content of "x" and "y" to the allocated corresponding device variables.

In "@jexitdata", users can copy back data from the device using "updatefrom" clause. "deallocate" clause deallocates device memory allocated for "x", "y", and "z".

"@jdecel" directive notifies Jai that the user will not use current accelerator context anymore.

You may notice that the Jai usage for fortran_openacc framework has similarity to fortran framework usage shown above. In fact, you can use the same code in fortran_openacc case for supporting not only fortran_openacc but also fortran if you switch "@jaccel" with proper information of framework and compile as shown below.

To use fortran_openacc
```julia
@jaccel myaccel2 framework(fortran_openacc) compile("ftn -h acc,noomp -fPIC -shared")
```

To use fortran
```julia
@jaccel myaccel2 framework(fortran) compile("gfortran -fPIC -shared")
```

In case of fortram framework, "@jenterdata" and "@jexitdata" silently exit without doing any work. Therefore, user can maintain the same Jai code for supporting multiple acceleration frameworks.

## Questions and Suggestions

Usage questions and suggestions can be posted on the [issue](https://github.com/grnydawn/AccelInterfaces.jl/issues).
