# Getting-Started

AccelInterfaces.jl(**Jai**) focuses on reusing Fortran and C/C++ codes, especially for large-scale simulation applications, within Julia. **Jai** does not limit its support to specific languages or programming frameworks, as long as the code can be compiled as a shared library. Currently, **Jai** supports the basic capabilities of Fortran, C/C++, Fortran OpenMP, Fortran OpenACC, C++ OpenMP, CUDA, as well as HIP.

## Installation

**Jai** can be installed using the Julia package manager.
From the Julia REPL, run

```
pkg> add AccelInterfaces
```

## Preparing a Fortran/C/C++ code to be called from Julia

First, we need to prepare a Fortran/C/C++("embedded") code that will be called from Julia main. The embedded code is the body part of Fortran/C/C++ function without a function signature and a functon end marker(such as "end function" in Fortran and "}"in C/C++). Jai will use the embedded code to generate a shared library after adding proper function signature based on input variables in Julia.

User can specify the function body code using Julia string that contains the code, or a Julia string that points to Jai Kernel File(.knl) in simple text format. The following Julia code calculates a vector sum, whose main algorithm is written in Fortran.

```julia
# main.jl

using AccelInterfaces

kernel_text = """
[fortran]
    INTEGER i

    DO i=LBOUND(x, 1), UBOUND(x, 1)
        z(i) = x(i) + y(i)
    END DO
"""
```

The "kernel\_text" string contains a Fortran DO loop that actually calculates the vector sum. The header at the top of the string specifies that the code contains both of Fortran code.

## Annotating Julia main code with Jai macros

Once "embedded" code is ready as explained in previous section, we need to use Jai macros to drive the execution of the embedded code.

```julia
    # continued from previous Julia code

    const N = 10
    x = fill(1, N)
    y = fill(2, N)
    z = fill(0, N)
    answer = fill(3, N)

    @jaccel

    @jkernel kernel_text mykernel framework(fortran="gfortran -fPIC -shared")

    @jlaunch mykernel input(x, y)  output(z)

    @jdecel

    @assert z == answer
```

#### Creates a Jai accelerator context
The @jaccel directive creates a Jai accelerator context.

#### Creates a Jai kernel context
The @jkernel directive creates a Jai kernel context. The user must specify the string of the kernel, as in this example. Alternatively, the user can provide Jai with a path string to a text file that contains the kernel. To identify the kernel context, we use the literal name mykernel.

The framework clause specifies the kind of acceleration, which in this example is Fortran. The user can provide Jai with the actual compiler command line to generate a shared library. The command line should include the compiler and all compiler flags, except the -o flag, which specifies the name of the output file and the path to the input source file.

#### Launches a kernel
The first argument to the @jlaunch directive is the name of the kernel context used in the @jkernel directive. The user then adds the names of variables to the input and output clauses accordingly. However, it is important to note that you should only use simple variable names for inputs and outputs to/from the kernel in the @jlaunch directive. For example, you cannot write something like this:
```julia
@jlaunch mykernel input(x+1, func(y)) output(z::Vector) # Jai Syntax Error
```

#### Remove a Jai accelerator context
Lastly, "@jdecel" is used to declare the end of the Jai accelerator context.

# Running the Julia main code

Run the Julia main. During the first run, Jai will genrate a shared library and load the shared library and finally make a call to the function in the shared library. This process is done automatically and takes some time at the first run. Howver, once the process is finished with success, the generated shared library is cached and will be loaded immediately unless there is no change.

```bash
> julia main.jl
```

## Further readings

[Examples](@ref) : Jai examples including using GPU
