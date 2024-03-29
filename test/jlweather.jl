    jlweather_kernel_text = """

[fortran]
INTEGER i, j, k

DO k=LBOUND(Y, 3), UBOUND(Y, 3)
    DO j=LBOUND(Y, 2), UBOUND(Y, 2)
        DO i=LBOUND(Y, 1), UBOUND(Y, 1)
            !Z(i, j-1, k-2) = X(i-2, j-1, k) + Y(i, j, k) * val
            Z(i, j, k) = X(i, j, k) + Y(i, j, k) * val
        END DO
    END DO
END DO

[hip]
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    Z[k][j][i] = X[k][j][i] + Y[k][j][i] * val;
"""


_DEV_ALLOC = true

function _perform(X, Y, Z, val)

    @jenterdata hipacc updateto(X, Y) enable_if(_DEV_ALLOC)

    @jlaunch mykernel hipacc input(X, Y, val) output(Z) hip(threads=(SHAPE,1), enable_if=_DEV_ALLOC)

    @jexitdata hipacc updatefrom(Z)  enable_if(_DEV_ALLOC) async
end

function jlweather_test()



    @jaccel hipacc

    #@jkernel jlweather_kernel_text mykernel hipacc framework(hip=hip_compile, fortran=fort_compile)
    @jkernel jlweather_kernel_text mykernel hipacc framework(fortran=fort_compile)
#    @jdiff hipacc fort_impl(DEV_ALLOC=false, X=1) hip_impl(DEV_ALLOC=true) begin

    X = fill(1::Int64, SHAPE)
    Y = fill(2::Int64, SHAPE)
    Z = fill(1::Int64, SHAPE)


	val = 2

    for idx in 1:30
        _perform(X, Y, Z, val)

        X .= X .* 2
    end

#    _X = fill(1::Int64, SHAPE)
#    X  = OffsetArray(_X, -1:(SHAPE[1]-2), 0:(SHAPE[2]-1), 1:SHAPE[3])
#    Y = fill(2::Int64, SHAPE)
#    _Z = fill(1::Int64, SHAPE)
#    Z  = OffsetArray(_Z, 1:SHAPE[1], 0:(SHAPE[2]-1), -1:(SHAPE[3]-2))
#
#    val = 2
#    sum1 = 0
#    sum2 = 0
#
#    @jenterdata hipacc alloc(X, Y, Z) enable_if(_DEV_ALLOC)
#
#    for idx in 1:30
#        c = Z
#
#        if idx % 2 == 0
#            a = X
#            b = Y
#           sum2 = sum2 + sum(_X .+ Y .* val) 
#        else
#            #a = X
#            #b = Y
#            a = Y
#            b = X
#           sum2 = sum2 + sum(Y .+ _X .* val) 
#        end
#
#        _perform(a, b, c, val)
#
#        sum1 = sum1 + sum(c)
#
#        X .= X .* 2
#    end

    @jexitdata hipacc delete(X, Y, Z) enable_if(_DEV_ALLOC) async

    @jwait hipacc

    #@test sum1 == sum2


#    end

    @jdecel hipacc


end
