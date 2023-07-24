# Jai: Julia accelerator interfaces

[![Build Status](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/grnydawn/AccelInterfaces.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Jai** is an accelerator programming interfaces for [Julia](http://julialang.org/) programmers.

**Jai** focuses on reusing Fortran and C/C++ codes, especially for large-scale simulation applications, within Julia. Jai does not limit its support to specific languages or programming frameworks, as long as the code can be compiled as a shared library. Currently, Jai supports the basic capabilities of Fortran, C/C++, Fortran OpenMP, Fortran OpenACC, C++ OpenMP, CUDA, as well as HIP.

This package is still in the early stages of development. Please use this package at your own risk.

## Package features

- Provides Julia users with an OpenMP-like macro interface to run CPU and GPU code.
- Automatically generates a shared library of pre-existing Fortran/C/C++ code so that it can be called from Julia.
- Provides a simple interface to exchange data between Julia Arrays and GPU memory.
- Allows different CPU and GPU programming frameworks to coexist within an application.
- Boosts the performance of original code through just-in-time compilation.

## Installation

```julia
Pkg.add("AccelInterfaces")
```

## Quickstart

The following Julia code calculates a vector sum, whose main algorithm is written in Fortran.

```julia
using AccelInterfaces

kernel_text = """
[fortran, fortran_openacc]
    INTEGER i

    !\$acc parallel loop
    DO i=LBOUND(x, 1), UBOUND(x, 1)
        z(i) = x(i) + y(i)
    END DO
    !\$acc end parallel loop
"""

    const N = 10
    x = fill(1, N)
    y = fill(2, N)
    z = fill(0, N)
    answer = fill(3, N)

    @jaccel

    @jkernel kernel_text mykernel1 framework(fortran="gfortran -fPIC -shared")

    @jlaunch mykernel1 input(x, y)  output(z)

    @assert z == answer

```

### Using Jai for Fortran(CPU) application

#### Specifies a kernel
The "kernel_text" string contains a Fortran DO loop that actually calculates the vector sum. OpenACC annotations surround the DO loop, and the header at the top of the string specifies that the code contains both of Fortran and Fortran OpenACC code. Users can select one of fortran or fortran_openacc using the framework clause of the @jaccel Jai directive, which is explained below.

#### Creates a Jai accelerator context
The @jaccel directive creates a Jai accelerator context.

#### Creates a Jai kernel context
The @jkernel directive creates a Jai kernel context. The user must specify the string of the kernel, as in this example. Alternatively, the user can provide Jai with a path string to a text file that contains the kernel. To identify the kernel context, we use the literal name mykernel1.

The framework clause specifies the kind of acceleration, which in this example is Fortran. The user can provide Jai with the actual compiler command line to generate a shared library. The command line should include the compiler and all compiler flags, except the -o flag, which specifies the name of the output file and the path to the input source file.

#### Launches a kernel
The first argument to the @jlaunch directive is the name of the kernel context used in the @jkernel directive. The user then adds the names of variables to the input and output clauses accordingly. However, it is important to note that you should only use simple variable names for inputs and outputs to/from the kernel in the @jlaunch directive. For example, you cannot write something like this:
```julia
@jlaunch mykernel1 input(x+1, func(y)) output(z::Vector) # Jai Syntax Error
```

### Using Jai for Fortran OpenACC(GPU) application

NOTE: To run the Fortran OpenACC case, copy the following code lines at the end of the previous example.

```julia
    fill!(z, 0)

    @jkernel kernel_text mykernel2 framework(fortran_openacc="ftn -h acc,noomp -fPIC -shared")

    @jenterdata alloc(x, y, z) updateto(x, y)

    @jlaunch mykernel2 input(x, y)  output(z)

    @jexitdata updatefrom(z) delete(x, y, z)

    @jdecel

    @assert z == answer
```
#### Specifies a kernel
The Fortran OpenACC code shares most of the code with the above Fortran example, with the exception of additional lines for OpenACC annotations. To use fortran_openacc, the user can simply add the name fortran_openacc to the header of the kernel string, as shown in the "kernel_text" variable in the example.

#### Creates a Jai kernel context
To compile the example code for Fortran OpenACC, the framework clause in the @jkernel macro must contain the compile string for OpenACC arrays.

#### Allocate GPU memory and copy data from Julia Arrays to GPU memory
The @jenterdata directive is used to allocate GPU memory and copy data from CPU to GPU. Once the user adds Julia variable names, Jai uses the data movement API according to the framework used, OpenACC in this case.

#### Launches a kernel
The same as the above Fortran example. However, some frameworks such as CUDA may require additional information, including kernel launch configuration.

#### Copy data from GPU memory to Julia Arrays and deallocate GPU memory
The @jexitdata directive is used to deallocate GPU memory and copy data from GPU to CPU. Once the user adds Julia variable names, Jai uses the data movement API according to the framework used, OpenACC in this case.

#### Remove a Jai accelerator context
Lastly, "@jdecel" is used to declare the end of the Jai accelerator context.


## Questions and Suggestions

Usage questions and suggestions can be posted on the [issue](https://github.com/grnydawn/AccelInterfaces.jl/issues).

[//]: # (generate docs: julia --project --color=yes docs/make.jl)
