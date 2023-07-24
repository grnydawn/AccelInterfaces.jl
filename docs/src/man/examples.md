# Examples

## Simple OpenAcc example

### Preparing an OpenACC code

User can specify the function body code using Julia string that contains the code, or a Julia string that points to Jai Kernel File(.knl) in simple text format. The following Julia code calculates a vector sum, whose main algorithm is written in Fortran OpenACC.

```julia
# main.jl

using AccelInterfaces

kernel_text = """
[fortran_openacc]
    INTEGER i

    !\$acc parallel loop 
    DO i=LBOUND(x, 1), UBOUND(x, 1)
        z(i) = x(i) + y(i)
    END DO
    !\$acc end parallel loop
"""
```

The "kernel\_text" string contains a Fortran DO loop that actually calculates the vector sum. OpenACC annotations surround the DO loop, and the header at the top of the string specifies that the code contains both of Fortran OpenACC code.

### Annotating Julia main code with Jai macros

Once "embedded" code is ready as explained in previous section, we need to use Jai macros to drive the execution of the embedded code.

```julia
    # continued from previous Julia code

    const N = 10
    x = fill(1, N)
    y = fill(2, N)
    z = fill(0, N)
    answer = fill(3, N)

    @jkernel kernel_text mykernel framework(fortran_openacc="ftn -h acc,noomp -fPIC -shared")

    @jenterdata alloc(x, y, z) updateto(x, y)

    @jlaunch mykerne input(x, y)  output(z)

    @jexitdata updatefrom(z) delete(x, y, z)

    @jdecel

    @assert z == answer
```

#### Creates a Jai accelerator context
The @jaccel directive creates a Jai accelerator context.

#### Creates a Jai kernel context
The @jkernel directive creates a Jai kernel context. As a first caluse of @jkernel, users should specify the Jai Kernel through a Julia string as in this example. Alternatively, the user can provide Jai with a path string to a text file that contains the kernel. To identify the kernel context, we use the literal name mykernel.

The framework clause specifies the kind of acceleration, which in this example is Fortran OpenACC. The user can provide Jai with the actual compiler command line to generate a shared library. The command line should include the compiler and all compiler flags, except the -o flag, which specifies the name of the output file and the path to the input source file.

#### Allocate GPU memory and copy data from Julia Arrays to GPU memory
The @jenterdata directive is used to allocate GPU memory and copy data from CPU to GPU. Once the user adds Julia variable names, Jai uses the data movement API according to the framework used, OpenACC in this case.

#### Launches a kernel
The first argument to the @jlaunch directive is the name of the kernel context used in the @jkernel directive. The user then adds the names of variables to the input and output clauses accordingly. However, it is important to note that you should only use simple variable names for inputs and outputs to/from the kernel in the @jlaunch directive. For example, you cannot write something like this:
```julia
@jlaunch mykernel input(x+1, func(y)) output(z::Vector) # Jai Syntax Error
```
#### Copy data from GPU memory to Julia Arrays and deallocate GPU memory
The @jexitdata directive is used to deallocate GPU memory and copy data from GPU to CPU. Once the user adds Julia variable names, Jai uses the data movement API according to the framework used, OpenACC in this case.

#### Remove a Jai accelerator context
Lastly, "@jdecel" is used to declare the end of the Jai accelerator context.

### Running the Julia main code

Run the Julia main. During the first run, Jai will genrate a shared library and load the shared library and finally make a call to the function in the shared library. This process is done automatically and takes some time at the first run. Howver, once the process is finished with success, the generated shared library is cached and will be loaded immediately unless there is no change.

```bash
> julia main.jl
```

## Further readings

[jlweather](@ref) : Jai implementations of [miniWeather](https://github.com/mrnorman/miniWeather)

