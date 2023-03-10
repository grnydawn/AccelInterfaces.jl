# fortran.jl: implement functions for Fortran framework

FORTRAN_TEMPLATE_MODULE = """
MODULE {modname}
USE, INTRINSIC :: ISO_C_BINDING

{specpart}

CONTAINS

{subppart}

END MODULE
"""

FORTRAN_TEMPLATE_FUNCTION = """
INTEGER (C_INT64_T) FUNCTION {prefix}{name}({dargs}) BIND(C, name="{prefix}{name}")
USE, INTRINSIC :: ISO_C_BINDING

{specpart}

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

{execpart}

{prefix}{name} = JAI_ERRORCODE

END FUNCTION
"""

function gencode_accel(
        frame       ::JAI_TYPE_FORTRAN,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS
    ) :: String

    fnames = ("get_num_devices", "get_device_num", "set_device_num",
                  "device_init", "device_fini", "wait")

    specs = Vector{String}(undef, length(fnames))
    funcs = Vector{String}(undef, length(fnames))

    for (i, name) in enumerate(fnames)
        specs[i] = "PUBLIC $(prefix)$(name)"
        dargs     = "buf"
        specpart = "INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: $(dargs)"
        execpart = ""
        funcs[i] = jaifmt(FORTRAN_TEMPLATE_FUNCTION, prefix=prefix, name=name, dargs=dargs,
                          specpart=specpart, execpart=execpart)
    end

    modname  = "mod_" * prefix * "accel"
    specpart = join(specs, "\n")
    subppart = join(funcs, "\n\n")

    return jaifmt(FORTRAN_TEMPLATE_MODULE, modname=modname, specpart=specpart,
                  subppart=subppart)
end


function gencode_data(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS
    ) :: String

    apiname  = JAI_MAP_API_FUNCNAME[apitype]
    funcname = prefix * apiname

    argnames = Vector{String}(undef, length(args))
    typedecls   = Vector{String}(undef, length(args))

    # merge args
    for (i, (var, vname, vinout, vshape, voffset)) in enumerate(args)
        argnames[i] = vname
        typedecls[i] = "INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: " * vname
    end

    dargs = join(argnames, ", ")
    funcspecpart = join(typedecls, "\n")
    funcexecpart = ""

    modsubppart = jaifmt(FORTRAN_TEMPLATE_FUNCTION, prefix=prefix, name=apiname,
                      dargs=dargs, specpart=funcspecpart, execpart=funcexecpart)

    modname  = "mod_" * funcname
    modspecpart = "PUBLIC " * funcname

    return jaifmt(FORTRAN_TEMPLATE_MODULE, modname=modname, specpart=modspecpart,
                  subppart=modsubppart)
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


function genslib_data(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        workdir     ::String,
        args        ::JAI_TYPE_ARGS
    ) :: Ptr{Nothing}

    code = gencode_data(frame, apitype, prefix, args)

    srcname = prefix * "data.F90"
    outname = prefix * "data." * dlext

    slibpath = compile_code(frame, code, srcname, outname, workdir)

    return load_sharedlib(slibpath)

end

