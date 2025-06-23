import tensorrt as trt
import numpy as np
import time
import onnx
import ctypes
from cuda import cuda
from cuda import cudart
import pandas as pd
from typing import List
from typing import Optional

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def calculate_latency(latency_list: list) -> float:
        """
        Calculates the latency of a measurement list as the mode of the distribution
        where the upper and lower 25% quantiles are cut off.

        Args:
            latency_list (list): List of latency values.

        Returns:
            float: Calculated latency value.
        """
        timing_df = pd.DataFrame(latency_list, columns=["latency"])

        Q1 = timing_df["latency"].quantile(0.25)
        Q3 = timing_df["latency"].quantile(0.75)
        median = timing_df["latency"].median()

        # removing outliers
        timing_df = timing_df[
            ~(
                (timing_df < (median - (((median - Q1) + 1) ** 2)))
                | (timing_df > ((((Q3 - median) + 1) ** 2) + median))
            ).any(axis=1)
        ]

        # removing top/bot 15%
        latency = sorted(timing_df["latency"], reverse=True)
        num_rows_to_drop = int(len(latency) * 0.15)
        latency = latency[num_rows_to_drop:][::-1][num_rows_to_drop:]
        avg = sum(latency) / len(latency)

        return round(avg, 5)

def free_buffers(
    inputs,
    outputs,
    stream: cudart.cudaStream_t,
):
    """
    Frees the resources allocated in allocate_buffers.

    Args:
        inputs (List[HostDeviceMem]): List of input buffers.
        outputs (List[HostDeviceMem]): List of output buffers.
        stream (cudart.cudaStream_t): CUDA stream.
    """
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))
class HostDeviceMem:
    """
    Taken from https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common.py
    Pair of host and device memory, where the host memory is wrapped in a numpy array

    Args:
    size (int): Size of the data.
    dtype (np.dtype): Data type of the data.
    """

    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        """
        Returns the host data.

        Returns:
            np.ndarray: Host data.
        """
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        """
        Sets the host data.

        Args:
            arr (np.ndarray): Array to set as host data.

        Raises:
            ValueError: If the array size is larger than the host memory size.
        """
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} \
                into host memory of size {self.host.size}"
            )
        np.copyto(self.host[: arr.size], arr.flat, casting="safe")

    @property
    def device(self) -> int:
        """
        Returns the device data.

        Returns:
            int: Device data.
        """
        return self._device

    @property
    def nbytes(self) -> int:
        """
        Returns the number of bytes in the memory.

        Returns:
            int: Number of bytes.
        """
        return self._nbytes

    def __str__(self):
        """
        Returns a string representation of the HostDeviceMem object.

        Returns:
            str: String representation.
        """
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        """
        Returns a string representation of the HostDeviceMem object.

        Returns:
            str: String representation.
        """
        return self.__str__()

    def free(self):
        """
        Frees the host and device memory.
        """
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

def build_engine(onnx_model_path, fp16):
    # Create a TensorRT logger
    logger = trt.Logger()
    runtime = trt.Runtime(logger)

    # get onnx model
    model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)
    serialized_onnx_model = model_onnx.SerializeToString()

    # Create a builder
    builder = trt.Builder(logger)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    parser = trt.OnnxParser(network, logger)
    config.max_workspace_size = 1 << 32  # sets the workspace size, 1 << 32 = 4GB
    builder.max_batch_size = 20

    # Load the ONNX model
    #with open(onnx_model_path, 'rb') as model:
    if not parser.parse(serialized_onnx_model):
        # raise error if parsing not successful
        parser_errors = ""
        for error in range(parser.num_errors):
            parser_errors = f"  {parser_errors}\n{parser.get_error(error)}"
        raise RuntimeError(
            f"Parsing of ONNX file to TensorRT engine \
            failed with the errors:\n{parser_errors}"
        )
    serialized_engine = builder.build_serialized_network(network, config)
    # Build the engine
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

def infer(engine, input_data):
    # Create a context for inference
    context = engine.create_execution_context()

    # Allocate device memory
    d_input = np.array(input_data, dtype=np.float32)
    d_output = np.empty(engine.get_binding_shape(1), dtype=np.float32)  # Assuming output is at index 1

    # Perform inference
    context.execute_v2(bindings=[d_input.ctypes.data, d_output.ctypes.data])
    return d_output

def check_cuda_err(err):
    """
    Taken from https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common.py
    """
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}")
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Runtime Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")
    
def cuda_call(call):
    """
    Taken from https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common.py
    """
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

def benchmark(onnx_model_path, num_runs=100, fp16=True):
    # Build the TensorRT engine
    engine = build_engine(onnx_model_path, fp16=fp16)
    if engine is None:
        print("Failed to create engine.")
        return
    avg_latency, timing_list, out_dict = execute(engine)
    print(f"Average Latency: {avg_latency:.5f} ms")
    return avg_latency


def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    """
    Taken from
    https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common.py
    Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    If engine uses dynamic shapes, specify a profile
    to find the maximum input & output size.

    Args:
        engine (trt.ICudaEngine): TensorRT engine.
        profile_idx (Optional[int], optional): Profile index for dynamic shapes.

    Returns:
        tuple: (inputs, outputs, bindings, stream)
                inputs: List of input buffers.
                outputs: List of output buffers.
                bindings: List of device bindings.
                stream: CUDA stream.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = (
            engine.get_tensor_shape(binding)
            if profile_idx is None
            else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        )
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(
                f"Binding {binding} has dynamic shape, "
                + "but no profile was specified."
            )
        size = trt.volume(shape)
        if engine.has_implicit_batch_dimension:
            size *= engine.max_batch_size
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        # dtype = trt.nptype(engine.get_tensor_dtype(binding))

        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings, stream

def execute(engine):
    context = engine.create_execution_context()
    print("Created execution context")
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print("Created input/output buffers")
    timing_list = []
    start_times = []
    out_dict = {}

    for idx in range(1000):
        start_times.append(time.time_ns())
        start = time.perf_counter()
        # copy inputs to device
        transfer_kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [
            cuda_call(
                cudart.cudaMemcpyAsync(
                    inp.device, inp.host, inp.nbytes, transfer_kind, stream
                )
            )
            for inp in inputs
        ]
        # start model execution
        context.execute_async_v2(bindings=bindings, stream_handle=stream)
        # copy results from device to host
        transfer_kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [
            cuda_call(
                cudart.cudaMemcpyAsync(
                    out.host, out.device, out.nbytes, transfer_kind, stream
                )
            )
            for out in outputs
        ]
        # synchronize cuda calls
        cuda_call(cudart.cudaStreamSynchronize(stream))
        end = time.perf_counter()
        timing_list.append((end - start) * 1e3)

    free_buffers(inputs, outputs, stream)
    avg_latency = calculate_latency(timing_list)
    return avg_latency, timing_list, out_dict

if __name__ == "__main__":
    # Specify the path to your ONNX model
    onnx_model_path = 'model.onnx'  # Change this to your model path

    # Create dummy input data (adjust shape as needed)
    input_shape = (1, 3, 224, 224)  # Example shape for an image input
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Run the benchmark
    benchmark(onnx_model_path, input_data)
