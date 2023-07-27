# Julia AccelInterfaces.jl (**Jai**)

*Accelerator programming interfaces for Julia programmers*

A package for reusing Fortran/C/C++ codes of large-scale simulations in Julia.

## Package Features

- Provides Julia users with an OpenMP-like macro interface to run CPU and GPU code.
- Automatically generates a shared library of pre-existing Fortran/C/C++ code so that it can be called from Julia.
- Provides a simple interface to exchange data between Julia Arrays and GPU memory.
- Allows different CPU and GPU programming frameworks to coexist within an application.
- Boosts the performance of original code through just-in-time compilation.

!!! warning

    This package is in the early stages of development. Please use it at your own risk.

The [Getting-Started](@ref) provides a tutorial explaining how to get started using AccelInterfaces.

The [Examples](@ref) shows a Jai example for OpenACC.

[jlweather demo](@ref) is a Julia port of [miniWeather](https://github.com/mrnorman/miniWeather) using [Jai](https://github.com/grnydawn/AccelInterfaces.jl).

See [Jai API](@ref) for the explations of Jai macros.
