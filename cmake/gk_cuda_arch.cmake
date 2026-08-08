# Choosing which CUDA architectures the device code is compiled for.
#
# This exists because CMake's `native` is not what it sounds like. CMake works
# out CMAKE_CUDA_ARCHITECTURES_NATIVE at the time it detects the compiler, and
# it filters what it finds against its own built-in table of architectures. The
# table belongs to CMake, not to the toolkit, so a CMake older than a GPU drops
# that GPU silently: CMake 3.30 knows nothing past 90, and on a machine with an
# RTX 4050 and an RTX 5090 it reports exactly `89-real`. Nothing warns. The
# build succeeds. Every kernel launched on the 5090 then fails at run time with
# "no kernel image is available for execution on the device", which is only
# reachable at all once a second device is in play - so it reads as a
# multi-GPU bug rather than a build one.
#
# The `-real` in that value is the second half of the problem. It means SASS
# only, with no PTX embedded, so there is nothing for the driver to JIT and an
# unrecognised device has no fallback at all.
#
# So we ask the two authorities directly instead: nvcc for what it can target,
# and the driver for what is actually installed. The result always carries PTX
# for the newest architecture in it, so a device newer than anything present at
# build time JITs rather than failing.

# Architectures this nvcc can generate code for, as bare integers.
function(_gk_nvcc_architectures out_var)
    set(${out_var} "" PARENT_SCOPE)

    if (NOT CMAKE_CUDA_COMPILER)
        return()
    endif()

    execute_process(
        COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-code
        OUTPUT_VARIABLE  gpu_code
        ERROR_VARIABLE   gpu_code_err
        RESULT_VARIABLE  gpu_code_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (NOT gpu_code_result EQUAL 0)
        return()
    endif()

    string(REGEX MATCHALL "sm_([0-9]+)" matches "${gpu_code}")

    set(archs "")
    foreach (match IN LISTS matches)
        string(REGEX REPLACE "sm_" "" arch "${match}")
        list(APPEND archs ${arch})
    endforeach()

    list(REMOVE_DUPLICATES archs)
    set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

# Compute capabilities of the GPUs attached to this machine, as bare integers.
#
# nvidia-smi is asked rather than CUDA, because it answers without compiling or
# running anything and it reports the capability of every device rather than
# just the one CUDA would pick as default. It is also the piece most likely to
# be present: a container built for GPU work has the driver utilities long
# before it has a working nvcc.
function(_gk_installed_architectures out_var)
    set(${out_var} "" PARENT_SCOPE)

    find_program(GK_NVIDIA_SMI
        NAMES nvidia-smi nvidia-smi.exe
        HINTS "$ENV{SystemRoot}/System32" "$ENV{ProgramFiles}/NVIDIA Corporation/NVSMI"
        DOC   "nvidia-smi, used to detect which GPUs this machine has")
    mark_as_advanced(GK_NVIDIA_SMI)

    if (NOT GK_NVIDIA_SMI)
        return()
    endif()

    execute_process(
        COMMAND "${GK_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE  caps
        ERROR_VARIABLE   caps_err
        RESULT_VARIABLE  caps_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (NOT caps_result EQUAL 0)
        return()
    endif()

    # "8.9" -> 89, "12.0" -> 120. The minor is a single digit on every part
    # shipped so far, which is the same assumption `major * 10 + minor` makes
    # everywhere else in the toolkit.
    string(REGEX MATCHALL "[0-9]+\\.[0-9]+" pairs "${caps}")

    set(archs "")
    foreach (pair IN LISTS pairs)
        string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)$" "\\1;\\2" parts "${pair}")
        list(GET parts 0 major)
        list(GET parts 1 minor)
        math(EXPR arch "${major} * 10 + ${minor}")
        list(APPEND archs ${arch})
    endforeach()

    list(REMOVE_DUPLICATES archs)
    set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

# CMake's own detection, with the -real/-virtual suffixes stripped. Used only
# when nvidia-smi could not be reached; it is the value this file exists to
# improve on, but an incomplete answer beats none.
function(_gk_cmake_native_architectures out_var)
    set(archs "")
    foreach (entry IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
        string(REGEX REPLACE "-(real|virtual)$" "" arch "${entry}")
        if (arch MATCHES "^[0-9]+$")
            list(APPEND archs ${arch})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES archs)
    set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()

# Picks the architecture list for a build that was not given one.
#
# The result is in CUDA_ARCHITECTURES form: `<n>-real` for each architecture
# present, and a bare `<max>` for the newest, which asks for its SASS *and* its
# PTX. That PTX is the forward-compatibility story - the driver JITs it for any
# device newer than the newest one seen here.
function(gk_choose_cuda_architectures out_var)
    set(${out_var} "" PARENT_SCOPE)

    _gk_nvcc_architectures(nvcc_archs)
    _gk_installed_architectures(installed_archs)

    set(detected_by "nvidia-smi")
    if (NOT installed_archs)
        _gk_cmake_native_architectures(installed_archs)
        set(detected_by "CMake's native detection")
    endif()

    if (NOT installed_archs)
        # A build machine with no GPU - a CI runner or a container producing a
        # wheel. Nothing can be detected, so target the family bases nvcc knows
        # about and let the PTX cover the rest. Deliberately not every
        # architecture in --list-gpu-code: that is a dozen copies of the device
        # code and a build to match, for parts that mostly no longer ship.
        set(fallback 75 80 86 89 90 100 120)
        set(installed_archs "")
        foreach (arch IN LISTS fallback)
            if (arch IN_LIST nvcc_archs)
                list(APPEND installed_archs ${arch})
            endif()
        endforeach()
        set(detected_by "no GPU found; portable default")
    endif()

    # Anything this nvcc cannot target is dropped, with a warning: it means the
    # toolkit is older than the card, and the only fix is a newer toolkit.
    set(archs "")
    set(unsupported "")
    foreach (arch IN LISTS installed_archs)
        if (NOT nvcc_archs OR arch IN_LIST nvcc_archs)
            list(APPEND archs ${arch})
        else()
            list(APPEND unsupported ${arch})
        endif()
    endforeach()

    if (unsupported)
        string(REPLACE ";" ", " unsupported_text "${unsupported}")
        message(WARNING
            "gk: this machine has a GPU of compute capability ${unsupported_text}, which "
            "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot generate code for. Those devices will "
            "run on JIT-compiled PTX if they can and be skipped if they cannot; a newer CUDA "
            "toolkit is the only way to give them native code.")
    endif()

    if (NOT archs)
        return()
    endif()

    # Highest wins the PTX. Compared as integers rather than sorted, because
    # list(SORT COMPARE NATURAL) needs CMake 3.18 and this file does not.
    set(newest 0)
    foreach (arch IN LISTS archs)
        if (arch GREATER newest)
            set(newest ${arch})
        endif()
    endforeach()

    if (CMAKE_VERSION VERSION_LESS 3.18)
        # No -real/-virtual before 3.18. A bare list asks for SASS and PTX for
        # every entry: larger than it needs to be, but correct.
        set(result "${archs}")
    else()
        set(result "")
        foreach (arch IN LISTS archs)
            if (arch EQUAL newest)
                list(APPEND result ${arch})   # SASS + PTX
            else()
                list(APPEND result ${arch}-real)
            endif()
        endforeach()
    endif()

    string(REPLACE ";" ", " archs_text "${archs}")
    message(STATUS
        "gk: CUDA architectures ${archs_text} (${detected_by}), with PTX for ${newest}")

    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# The bare integers behind a CUDA_ARCHITECTURES value, for reporting at run
# time. gk prints these when a device turns out to have no kernel image, which
# is the one moment somebody needs to know what the binary was built for.
function(gk_cuda_architectures_to_list architectures out_var)
    set(archs "")
    foreach (entry IN LISTS architectures)
        string(REGEX REPLACE "-(real|virtual)$" "" arch "${entry}")
        if (arch MATCHES "^[0-9]+$")
            list(APPEND archs ${arch})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES archs)
    string(REPLACE ";" " " archs_text "${archs}")
    set(${out_var} "${archs_text}" PARENT_SCOPE)
endfunction()
