# fortran.jl: implement functions for Fortran framework

function gencode_accel(frame)
end

function compile_code(frame, code)
end

function genslib_accel(
        frame       ::JAI_TYPE_FORTRAN,
        aname       ::String
    ) :: String

    slib = nothing

    code = gencode_accel(frame)

    slib = compile_code(frame, code)

    inargs, outargs, innames, outnames = [], [], [], []

    invoke_slibfunc(frame, slib, "ja_"*aname, inargs, outargs, innames, outnames)

    return slib

end

