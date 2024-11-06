# ccall.pyx

# Import Cython and types from standard Python
import numpy as np
from typing import List, Dict
cimport numpy as np
from libc.stdint cimport int64_t, uint8_t, uint32_t
from libc.stdlib cimport malloc, free
from libc.math cimport pow

from cython.parallel import prange, threadid
cimport cython
cimport openmp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ccall_bool(
        uint32_t [:, ::1] expons, 
        uint8_t [:, ::1] coeffs, 
        uint8_t [:, ::1] out, 
        uint8_t [::1] ones, 
        uint8_t [:, ::1] parameters, 
):
  
    cdef Py_ssize_t i, j, k, u, iloop, jloop, kloop, uloop, num_threads, thread_id

    iloop = expons.shape[0]
    jloop = expons.shape[1]
    kloop = ones.shape[0]
    uloop = out.shape[0]

    num_threads = openmp.omp_get_max_threads()
    cdef uint8_t[:, :, :] accum = np.zeros((num_threads, uloop, kloop), dtype=bool) 
    cdef uint8_t[:, ::1] term = np.empty((num_threads, kloop), dtype=bool)
    with cython.nogil, cython.parallel.parallel():  # Parallel block
        for i in prange(iloop, schedule='static'):
            thread_id = threadid()
            
            for k in range(kloop):
                term[thread_id, k] = ones[k]

            for j in range(jloop):
                for k in range(kloop):
                    term[thread_id, k] *= parameters[j, k] ** <unsigned int>expons[i, j]

            for u in range(uloop):
                for k in range(kloop):
                    accum[thread_id, u, k] += coeffs[i, u] * term[thread_id, k]

    with cython.nogil:
        for u in prange(uloop, schedule='static'):
            for k in range(kloop):
                for i in range(num_threads):
                    out[u, k] += accum[i, u, k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ccall_uint32(
        uint32_t [:, ::1] expons, 
        uint32_t [:, ::1] coeffs, 
        uint32_t [:, ::1] out, 
        uint32_t [::1] ones, 
        uint32_t [:, ::1] parameters, 
):
  
    cdef Py_ssize_t i, j, k, u, iloop, jloop, kloop, uloop, num_threads, thread_id

    iloop = expons.shape[0]
    jloop = expons.shape[1]
    kloop = ones.shape[0]
    uloop = out.shape[0]

    num_threads = openmp.omp_get_max_threads()
    cdef uint32_t[:, :, :] accum = np.zeros((num_threads, uloop, kloop), dtype=np.uint32) 
    cdef uint32_t[:, ::1] term = np.empty((num_threads, kloop), dtype=np.uint32)
    with cython.nogil, cython.parallel.parallel():  # Parallel block
        for i in prange(iloop, schedule='static'):
            thread_id = threadid()
            
            for k in range(kloop):
                term[thread_id, k] = ones[k]

            for j in range(jloop):
                for k in range(kloop):
                    term[thread_id, k] *= parameters[j, k] ** <unsigned int>expons[i, j]

            for u in range(uloop):
                for k in range(kloop):
                    accum[thread_id, u, k] += coeffs[i, u] * term[thread_id, k]

    with cython.nogil:
        for u in prange(uloop, schedule='static'):
            for k in range(kloop):
                for i in range(num_threads):
                    out[u, k] += accum[i, u, k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ccall_int(
        uint32_t [:, ::1] expons, 
        int64_t [:, ::1] coeffs, 
        int64_t [:, ::1] out, 
        int64_t [::1] ones, 
        int64_t [:, ::1] parameters, 
):
  
    cdef Py_ssize_t i, j, k, u, iloop, jloop, kloop, uloop, num_threads, thread_id

    iloop = expons.shape[0]
    jloop = expons.shape[1]
    kloop = ones.shape[0]
    uloop = out.shape[0]

    num_threads = openmp.omp_get_max_threads()
    cdef int64_t[:, :, :] accum = np.zeros((num_threads, uloop, kloop), dtype=np.int64) 
    cdef int64_t[:, ::1] term = np.empty((num_threads, kloop), dtype=np.int64)
    with cython.nogil, cython.parallel.parallel():  # Parallel block
        for i in prange(iloop, schedule='static'):
            thread_id = threadid()
            
            for k in range(kloop):
                term[thread_id, k] = ones[k]

            for j in range(jloop):
                for k in range(kloop):
                    term[thread_id, k] *= parameters[j, k] ** <unsigned int>expons[i, j]

            for u in range(uloop):
                for k in range(kloop):
                    accum[thread_id, u, k] += coeffs[i, u] * term[thread_id, k]

    with cython.nogil:
        for u in prange(uloop, schedule='static'):
            for k in range(kloop):
                for i in range(num_threads):
                    out[u, k] += accum[i, u, k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ccall_float(
        uint32_t [:, ::1] expons, 
        double [:, ::1] coeffs, 
        double [:, ::1] out, 
        double [::1] ones, 
        double [:, ::1] parameters, 
):
  
    cdef Py_ssize_t i, j, k, u, iloop, jloop, kloop, uloop, num_threads, thread_id

    iloop = expons.shape[0]
    jloop = expons.shape[1]
    kloop = ones.shape[0]
    uloop = out.shape[0]

    num_threads = openmp.omp_get_max_threads()
    cdef double[:, :, :] accum = np.zeros((num_threads, uloop, kloop), dtype=np.float64) 
    cdef double[:, ::1] term = np.empty((num_threads, kloop), dtype=np.float64)
    with cython.nogil, cython.parallel.parallel():  # Parallel block
        for i in prange(iloop, schedule='static'):
            thread_id = threadid()
            
            for k in range(kloop):
                term[thread_id, k] = ones[k]

            for j in range(jloop):
                for k in range(kloop):
                    term[thread_id, k] *= parameters[j, k] ** <unsigned int>expons[i, j]

            for u in range(uloop):
                for k in range(kloop):
                    accum[thread_id, u, k] += coeffs[i, u] * term[thread_id, k]

    with cython.nogil:
        for u in prange(uloop, schedule='static'):
            for k in range(kloop):
                for i in range(num_threads):
                    out[u, k] += accum[i, u, k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ccall_complex(
        uint32_t [:, ::1] expons, 
        complex [:, ::1] coeffs, 
        complex [:, ::1] out, 
        complex [::1] ones, 
        complex [:, ::1] parameters, 
):
  
    cdef Py_ssize_t i, j, k, u, iloop, jloop, kloop, uloop, num_threads, thread_id

    iloop = expons.shape[0]
    jloop = expons.shape[1]
    kloop = ones.shape[0]
    uloop = out.shape[0]

    num_threads = openmp.omp_get_max_threads()
    cdef complex[:, :, :] accum = np.zeros((num_threads, uloop, kloop), dtype=np.complex128) 
    cdef complex[:, ::1] term = np.empty((num_threads, kloop), dtype=np.complex128)
    with cython.nogil, cython.parallel.parallel():  # Parallel block
        for i in prange(iloop, schedule='static'):
            thread_id = threadid()
            
            for k in range(kloop):
                term[thread_id, k] = ones[k]

            for j in range(jloop):
                for k in range(kloop):
                    term[thread_id, k] *= parameters[j, k] ** <unsigned int>expons[i, j]

            for u in range(uloop):
                for k in range(kloop):
                    accum[thread_id, u, k] += coeffs[i, u] * term[thread_id, k]

    with cython.nogil:
        for u in prange(uloop, schedule='static'):
            for k in range(kloop):
                for i in range(num_threads):
                    out[u, k] += accum[i, u, k]


def ccall(
        np.ndarray[np.uint32_t, ndim=2] expons, 
        List[np.ndarray] coefficients, 
        np.ndarray ones, 
        Dict[str, np.ndarray] parameters, 
        shape,
):
    result_type = np.result_type(next(iter(parameters.values())).dtype, coefficients[0].dtype)   

    out = np.zeros(shape, dtype=result_type)
    ones = ones.astype(result_type)
    params = np.asarray(list(parameters.values()), dtype=result_type)
    coeffs = np.asarray(coefficients, dtype=result_type)

    if result_type == bool:
        ccall_int(expons, coeffs, out, ones, params)
    elif result_type == np.uint32:
        ccall_uint32(expons, coeffs, out, ones, params)
    elif result_type == np.int64:
        ccall_int(expons, coeffs, out, ones, params)
    elif result_type == np.float64:
        ccall_float(expons, coeffs, out, ones, params)
    elif result_type == np.complex128:
        ccall_complex(expons, coeffs, out, ones, params)

    return out
