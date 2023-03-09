# fortran.jl: implement functions for Fortran framework

function gencode_accel(
        frame       ::JAI_TYPE_FORTRAN,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS
    ) :: String

    return  """
MODULE mod_$(prefix)accel
USE, INTRINSIC :: ISO_C_BINDING

PUBLIC $(prefix)get_num_devices
PUBLIC $(prefix)get_device_num
PUBLIC $(prefix)set_device_num
PUBLIC $(prefix)device_init
PUBLIC $(prefix)device_fini
PUBLIC $(prefix)wait

CONTAINS

INTEGER (C_INT64_T) FUNCTION $(prefix)device_init(buf) BIND(C, name="$(prefix)device_init")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(prefix)device_init_ = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION $(prefix)device_fini(buf) BIND(C, name="$(prefix)device_fini")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(prefix)device_fini = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION $(prefix)get_num_devices(buf) BIND(C, name="$(prefix)get_num_devices")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

buf(1) = 1

$(prefix)get_num_devices = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION $(prefix)get_device_num(buf) BIND(C, name="$(prefix)get_device_num")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

buf(1) = 1

$(prefix)get_device_num = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION $(prefix)set_device_num(buf) BIND(C, name="$(prefix)set_device_num")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(IN) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(prefix)set_device_num = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION $(prefix)wait() BIND(C, name="$(prefix)wait")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(prefix)wait = JAI_ERRORCODE

END FUNCTION

END MODULE
"""
end

function compile_code(
        frame       ::JAI_TYPE_FORTRAN,
        code        ::String,
        srcname     ::String,
        outname     ::String,
        workdir     ::String
    )

    compile = "gfortran -fPIC -shared -g -ffree-line-length-none"

    return compile_code(code, compile, srcname, outname, workdir)
end

function genslib_accel(
        frame       ::JAI_TYPE_FORTRAN,
        prefix      ::String,               # prefix for libfunc names
        workdir     ::String,
        args        ::JAI_TYPE_ARGS
    ) :: Ptr{Nothing}

    code = gencode_accel(frame, prefix, args)

    srcname = prefix * "accel.F90"
    outname = prefix * "accel." * dlext

    slibpath = compile_code(frame, code, srcname, outname, workdir)

    slib = load_sharedlib(slibpath)

    # init device
    invoke_slibfunc(frame, slib, prefix * "device_init", args)

    return slib

end

