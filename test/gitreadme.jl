gitreadme_kernel_text = """
[fortran, fortran_openacc]

INTEGER i

!\$acc parallel loop 
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


function gitreadme_test()

    #@jaccel framework(fortran="gfortran -fPIC -shared")
    @jaccel

    @jkernel gitreadme_kernel_text mykernel1  framework(fortran="gfortran -fPIC -shared")

    @jlaunch mykernel1 input(x, y)  output(z)

    @assert z == answer

    fill!(z, 0)

    #@jaccel framework(fortran_openacc="ftn -h acc,noomp -fPIC -shared")
    #@jaccel 

    @jkernel gitreadme_kernel_text mykernel2 framework(fortran_openacc="ftn -h acc,noomp -fPIC -shared")

    @jenterdata alloc(x, y, z) updateto(x, y)

    @jlaunch mykernel2 input(x, y)  output(z)

    @jexitdata updatefrom(z) delete(x, y, z)

    @jdecel

    @assert z == answer
end
