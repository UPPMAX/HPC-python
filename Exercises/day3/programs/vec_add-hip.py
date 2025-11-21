__author__ = "Shucai Xiao, Advanced Micro Devices, Inc. <shucai.xiao@amd.com>"

import ctypes
import array
import random
import math

from hip import hip, hiprtc
from icache_flush import icache_flush

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]

    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
        ):
        raise RuntimeError(str(err))

    return result


def vec_scale(x_h, n, factor):
    source = b"""\
        extern "C" __global__ void scale_vector(float *vec, float factor, int n) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid < n) {
                vec[tid] *= factor;
            }
        }
    """

    prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"scale_vector", 0, [], []))

    progs = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(progs, 0))
    arch = progs.gcnArchName

    # print(f"Compiling kernel for {arch}")
    # print(f"cu_num = {progs.multiProcessorCount}")

    cflags = [b"--offload-arch="+arch]
    err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        raise RuntimeError(log.decode())

    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"scale_vector"))

    num_bytes = x_h.itemsize * len(x_h)
    x_d = hip_check(hip.hipMalloc(num_bytes))
    print(f"{hex(int(x_d))=}")
    hip_check(hip.hipMemcpy(x_d, x_h, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    block = hip.dim3(x=32)
    grid = hip.dim3(math.ceil(n/block.x))

    icache_flush()

    hip_check(hip.hipModuleLaunchKernel(
        kernel,
        *grid,
        *block,
        sharedMemBytes=0,
        stream=None,
        kernelParams=None,
        extra=(
            x_d,
            ctypes.c_float(factor),
            ctypes.c_int(n),
        )
        )
    )

    x_out = array.array("f", [random.random() for i in range(0, n)])

    hip_check(hip.hipMemcpy(x_out, x_d, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    hip_check(hip.hipFree(x_d))
    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    return x_out


def main():
    factor = 1.23
    n = 100
    x_h = array.array("f", [random.random() for i in range(0, n)])
    x_out = vec_scale(x_h, n, factor)
    x_expected = [a * factor for a in x_h]

    for i, x_out_i in enumerate(x_out):
        if not math.isclose(x_out_i, x_expected[i], rel_tol=1e-6):
            raise RuntimeError(f"values do not match, {x_out[i] = } vs. {x_expected[i]=}, {i=}")

    print("ok")


if __name__ == "__main__":
    main()
