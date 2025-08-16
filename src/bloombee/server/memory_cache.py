"""
A pytorch memory cache that can be allocated by ConnectionHandler (on cpu) and used over multiple calls to Runtime.

For now, the only purpose of this code is to ensure that allocated memory will be deleted properly.

"""
import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import time

import logging
from typing import AsyncContextManager, Dict, Optional, Sequence, Tuple, Union, Any, List

import async_timeout
import torch
import dataclasses
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger


from bloombee.data_structures import Handle
from transformers import PretrainedConfig

from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.misc import get_size_in_bytes
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.task import Task

from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice, DeviceType, general_copy
from bloombee.flexgen_utils.compression import TorchCompressedDevice, general_copy_compressed
from bloombee.flexgen_utils.utils import torch_dtype_to_np_dtype
import numpy as np
from types import SimpleNamespace

logger = get_logger(__name__)

# 创建专门的offloading调试logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, 
                 max_size_bytes: Optional[int], 
                 max_alloc_timeout: Optional[float] = None, 
                 policy: Optional[Policy] = None, 
                 block_config: Optional[PretrainedConfig] = None,
                 device: Any = None):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.max_alloc_timeout = max_alloc_timeout
        self._lock_metadata = mp.Lock()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=False)
        self._enqueued_size = mp.Value(ctypes.c_int64, 0, lock=True)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._allocated_tensors: Dict[Handle, Any] = {}
        self._handle_size_bytes: Dict[Handle, int] = {}
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._lock_acquire_memory = mp.Lock()
        self._memory_freed_event = mp.Event()
        

        # flexgen' offloading depends on the task parameter, we need to mock a temp task variable for memory allocation without changing the data structure of flexgen.
        self.mocked_task = Task(
                inputs=None,
                prompt_len=0,
                gen_len=2048,
                cut_gen_len=None,
                do_sample=False,
                temperature=0,
                stop=None,
                top_p=None,
        )
        self.allocation_policy = policy
        self.block_config = block_config
        self.device = device
        # Initialize decompress workspace if using compressed CPU backend
        try:
            if isinstance(self.device, TorchCompressedDevice) and getattr(self.device, 'base_device', None) is not None:
                if self.device.base_device.device_type == DeviceType.CPU:
                    self.device.init_attention_compute_workspace(self.block_config, self.mocked_task, self.allocation_policy)
        except Exception:
            # Workspace is an optimization; proceed without if unavailable
            pass
        # For disk/mixed staging
        self._underlying: Dict[Handle, Any] = {}
        self._cpu_stage: Dict[Handle, Any] = {}
        self._layout_meta: Dict[Handle, Tuple[bool, int, int, int, int]] = {}  # (is_key, B, H, S, D)
        self._dtype_by_handle: Dict[Handle, torch.dtype] = {}
        

    @property
    def current_size_bytes(self) -> int:
        return self._current_size.value

    @current_size_bytes.setter
    def current_size_bytes(self, value: int):
        self._current_size.value = value

    @property
    def enqueued_size_bytes(self) -> int:
        return self._enqueued_size.value

    @enqueued_size_bytes.setter
    def enqueued_size_bytes(self, value: int):
        self._enqueued_size.value = value

    @property
    def bytes_left(self) -> int:
        return self.max_size_bytes - self.current_size_bytes

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    async def _schedule_alloc(
        self, alloc_size: int, *descriptors: TensorDescriptor, timeout: Optional[float]
    ) -> Sequence[Handle]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """
        try:
            async with self._wait_for_free_memory(alloc_size, timeout):
                with self._lock_metadata:
                    handles = tuple(int(self.handle_counter) + i for i in range(len(descriptors)))
                    self.current_size_bytes += alloc_size
                    self.handle_counter += len(handles)  # note: this will eventually overflow and it is okay
                    self._pipe_send.send((handles, descriptors))
                    return handles
        except TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} (timeout={timeout})")

    @contextlib.asynccontextmanager
    async def _wait_for_free_memory(self, alloc_size: int, timeout: Optional[float]):
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        with self._enqueued_size.get_lock():
            self._enqueued_size.value += alloc_size
        allocated = False
        try:
            context_manager = async_timeout.timeout(timeout) if timeout != 0 else contextlib.AsyncExitStack()
            # contextlib.AsyncExitStack() is used as a null context here
            async with context_manager:
                if timeout == 0 and self.current_size_bytes + self.enqueued_size_bytes > self.max_size_bytes:
                    raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                async with enter_asynchronously(self._lock_acquire_memory):
                    if self.current_size_bytes + alloc_size > self.max_size_bytes:
                        if timeout == 0:
                            raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                        elapsed_time = time.perf_counter() - start_time
                        remaining_timeout = max(0.0, timeout - elapsed_time) if timeout is not None else None
                        await loop.run_in_executor(None, self._wait_until_available, alloc_size, remaining_timeout)

                allocated = True
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size
                yield
        except asyncio.TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} within {timeout} seconds")
        finally:
            if not allocated:
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size

    def _free(self, alloc_size: int, alloc_task: asyncio.Task):
        if alloc_task.exception() is not None:
            return
        handles = alloc_task.result()

        with self._lock_metadata:
            self._pipe_send.send((handles, None))  # signal runtime to free these handles
            self.current_size_bytes -= alloc_size
        self._memory_freed_event.set()

    def _wait_until_available(self, allocated_size: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        timeout = timeout if timeout != float("inf") else None
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.current_size_bytes + allocated_size > self.max_size_bytes:
            remaining_time = None if timeout is None else deadline - time.perf_counter()
            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                )
            self._memory_freed_event.clear()

    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        """
        Return one or more tensors previously allocated with allocate_cache,

        :note: This method is called by ModuleBackend in runtime: a single process with NO process parallelism.
        However, runtime may call use_cache concurrently with one or more connection handlers calling allocate_cache
        """
        assert os.getpid() == self.runtime_pid
        # note: this specific function is not concurrent, so you can safely allocate/offload/defragment data here

        # Fast path: simple FlexGen-backed allocation on a single TorchDevice (GPU/CPU), no compression/mixed/disk
        if isinstance(self.device, TorchDevice):
            # read creation/deletion requests from connection handlers
            while self._pipe_recv.poll():
                recv_handles, recv_data = self._pipe_recv.recv()
                if recv_data is not None:
                    assert len(recv_handles) == len(recv_data)
                    assert len(recv_handles) % 2 == 0, "KV handles must come in K/V pairs"
                    i = 0
                    while i < len(recv_handles):
                        key_handle, val_handle = recv_handles[i], recv_handles[i + 1]
                        key_descr, val_descr = recv_data[i], recv_data[i + 1]

                        # Infer B,H,D,S from key descriptor (shape = [B, H, D, S])
                        bsz, num_heads, head_dim, seq_len = key_descr.shape  # type: ignore

                        # Per-request task: set S exactly
                        local_task = dataclasses.replace(
                            self.mocked_task, prompt_len=int(seq_len), gen_len=1
                        )

                        # Per-shard config: ensure head_dim stays the same
                        fake_cfg = SimpleNamespace(
                            num_attention_heads=int(num_heads),
                            hidden_size=int(head_dim * num_heads),
                        )

                        # Per-shard policy: ensure batch size matches descriptor
                        local_policy = dataclasses.replace(self.allocation_policy, gpu_batch_size=int(bsz))

                        # Allocate FlexGen caches (S, B*H, D)
                        k_wrap, v_wrap = self.device.init_cache_one_gpu_batch(fake_cfg, local_task, local_policy)

                        # Map to torch views expected by the rest of pipeline
                        # Keys: [B, H, D, S], Values: [B, H, S, D]
                        k_view = k_wrap.data.view(seq_len, bsz, num_heads, head_dim).permute(1, 2, 3, 0)
                        v_view = v_wrap.data.view(seq_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)

                        self._allocated_tensors[key_handle] = k_view
                        self._allocated_tensors[val_handle] = v_view

                        # Track sizes for force_free accounting
                        self._handle_size_bytes[key_handle] = int(np.prod(key_descr.shape)) * get_size_in_bytes(key_descr.dtype)
                        self._handle_size_bytes[val_handle] = int(np.prod(val_descr.shape)) * get_size_in_bytes(val_descr.dtype)
                        self._dtype_by_handle[key_handle] = key_descr.dtype
                        self._dtype_by_handle[val_handle] = val_descr.dtype

                        logger.info(
                            f"OFFLOAD: FlexGen KV ready via init_cache_one_gpu_batch: B={bsz}, H={num_heads}, D={head_dim}, S={seq_len}"
                        )
                        i += 2
                else:
                    for handle in recv_handles:
                        if handle not in self._allocated_tensors:
                            logger.warning(
                                f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                            )
                        else:
                            logger.info(f"OFFLOAD: Freed KV tensor handle={handle}")
                        self._allocated_tensors.pop(handle, None)
                        self._handle_size_bytes.pop(handle, None)
                        self._dtype_by_handle.pop(handle, None)

            try:
                yield tuple(self._allocated_tensors[handle] for handle in handles)
            finally:
                # No sync needed for direct TorchDevice-backed views
                pass
            return

        # Fallback path: full offloading (disk/mixed/compressed) support
        # read creation/deletion requests from connection handlers
        while self._pipe_recv.poll():
            recv_handles, recv_data = self._pipe_recv.recv()
            if recv_data is not None:  # create new tensors
                assert len(recv_handles) == len(recv_data)
                assert len(recv_handles) % 2 == 0, "KV handles must come in K/V pairs"
                i = 0
                while i < len(recv_handles):
                    key_handle, val_handle = recv_handles[i], recv_handles[i + 1]
                    key_descr, val_descr = recv_data[i], recv_data[i + 1]

                    # Infer B,H,D,S from key descriptor (shape = [B, H, D, S])
                    bsz, num_heads, head_dim, seq_len = key_descr.shape  # type: ignore

                    # Per-request task/policy and minimal config for FlexGen allocation
                    local_task = dataclasses.replace(self.mocked_task, prompt_len=int(seq_len), gen_len=1)
                    fake_cfg = SimpleNamespace(
                        num_attention_heads=int(num_heads),
                        hidden_size=int(head_dim * num_heads),
                    )
                    local_policy = dataclasses.replace(self.allocation_policy, gpu_batch_size=int(bsz))

                    # Allocate underlying KV via FlexGen device API (works for Disk/Mixed/Compressed too)
                    k_wrap, v_wrap = self.device.init_cache_one_gpu_batch(fake_cfg, local_task, local_policy)

                    # Track meta and sizes
                    self._underlying[key_handle] = k_wrap
                    self._underlying[val_handle] = v_wrap
                    self._layout_meta[key_handle] = (True, int(bsz), int(num_heads), int(seq_len), int(head_dim))
                    self._layout_meta[val_handle] = (False, int(bsz), int(num_heads), int(seq_len), int(head_dim))
                    self._allocated_tensors[key_handle] = None
                    self._allocated_tensors[val_handle] = None

                    self._handle_size_bytes[key_handle] = int(np.prod(tuple(key_descr.shape))) * get_size_in_bytes(key_descr.dtype)
                    self._handle_size_bytes[val_handle] = int(np.prod(tuple(val_descr.shape))) * get_size_in_bytes(val_descr.dtype)
                    self._dtype_by_handle[key_handle] = key_descr.dtype
                    self._dtype_by_handle[val_handle] = val_descr.dtype

                    logger.info(
                        f"OFFLOAD: Prepared FlexGen KV storage (device={type(self.device).__name__}) for B={bsz}, H={num_heads}, D={head_dim}, S={seq_len}"
                    )
                    i += 2
            else:  # delete tensors by handle
                for handle in recv_handles:
                    if handle not in self._allocated_tensors:
                        logger.warning(
                            f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                        )
                    else:
                        logger.info(f"OFFLOAD: Freed KV tensor handle={handle}")
                    self._allocated_tensors.pop(handle, None)
                    self._handle_size_bytes.pop(handle, None)
                    self._underlying.pop(handle, None)
                    self._layout_meta.pop(handle, None)
                    self._cpu_stage.pop(handle, None)
                    self._dtype_by_handle.pop(handle, None)
        # Materialize CPU tensors for FlexGen-backed handles
        materialized: List[torch.Tensor] = []
        for handle in handles:
            if handle in self._underlying:
                is_key, B, H, S, D = self._layout_meta[handle]
                BH = B * H
                # Stage always in fp16 for compressed; otherwise follow original dtype
                if isinstance(self.device, TorchCompressedDevice):
                    stage_np_dtype = np.float16
                else:
                    torch_dtype = self._dtype_by_handle.get(handle, torch.float16)
                    stage_np_dtype = torch_dtype_to_np_dtype.get(torch_dtype, np.float16)
                cpu_stage = TorchDevice("cpu").allocate((S, BH, D), stage_np_dtype, pin_memory=True)
                try:
                    if isinstance(self.device, TorchCompressedDevice):
                        comp_cfg = getattr(self.allocation_policy, 'comp_cache_config', None)
                        # If compressed base is CPU, decompress directly; if DISK, stage via CPU compressed device
                        if self.device.base_device.device_type == DeviceType.CPU:
                            decompressed = self.device.decompress(self._underlying[handle])
                            cpu_stage.data.copy_(decompressed)
                            logger.info(f"OFFLOAD: COMPRESS copy-in (decompress, base=CPU) handle={handle} -> CPU stage")
                        else:
                            cpu_comp = TorchDevice("cpu").compressed_device
                            tmp_comp = cpu_comp.allocate((S, BH, D), np.float16, comp_config=comp_cfg, pin_memory=True)
                            general_copy_compressed(tmp_comp, None, self._underlying[handle], None)
                            decompressed = cpu_comp.decompress(tmp_comp)
                            cpu_stage.data.copy_(decompressed)
                            logger.info(f"OFFLOAD: COMPRESS copy-in (decompress via CPU) handle={handle} -> CPU stage")
                    else:
                        general_copy(cpu_stage, None, self._underlying[handle], None)
                except Exception as e:
                    logger.warning(f"OFFLOAD: copy-in failed for handle={handle}: {e}. Filling zeros.")
                    cpu_stage.data.zero_()
                if is_key:
                    # View-only transform to [B, H, D, S]; keep it as a view so in-place writes update cpu_stage
                    tensor = cpu_stage.data.view(S, B, H, D).permute(1, 2, 3, 0)
                else:
                    # View-only transform to [B, H, S, D]
                    tensor = cpu_stage.data.view(S, B, H, D).permute(1, 2, 0, 3)
                self._cpu_stage[handle] = cpu_stage
                self._allocated_tensors[handle] = tensor
                materialized.append(tensor)
            else:
                # It is possible that this handle was created for underlying only
                # Ensure presence; otherwise initialize a zero CPU tensor as a fallback
                tensor = self._allocated_tensors.get(handle)
                if tensor is None:
                    # Fallback zero tensor based on recorded layout
                    is_key, B, H, S, D = self._layout_meta[handle]
                    if is_key:
                        tensor = torch.zeros((B, H, D, S), dtype=self._dtype_by_handle.get(handle, torch.float16), device='cpu', pin_memory=True)
                    else:
                        tensor = torch.zeros((B, H, S, D), dtype=self._dtype_by_handle.get(handle, torch.float16), device='cpu', pin_memory=True)
                    self._allocated_tensors[handle] = tensor
                materialized.append(tensor)

        try:
            yield tuple(materialized)
        finally:
            for handle in handles:
                if handle in self._underlying and handle in self._cpu_stage:
                    try:
                        if isinstance(self.device, TorchCompressedDevice):
                            comp_cfg = getattr(self.allocation_policy, 'comp_cache_config', None)
                            cpu_comp = TorchDevice("cpu").compressed_device
                            compressed = cpu_comp.compress(self._cpu_stage[handle].data, comp_cfg)
                            general_copy_compressed(self._underlying[handle], None, compressed, None)
                            logger.info(f"OFFLOAD: Synced COMPRESSED KV handle={handle} back to {type(self.device.base_device).__name__}")
                        else:
                            general_copy(self._underlying[handle], None, self._cpu_stage[handle], None)
                            logger.info(f"OFFLOAD: Synced KV handle={handle} back to {type(self.device).__name__}")
                    except Exception as e:
                        logger.warning(f"OFFLOAD: copy-out failed for handle={handle}: {e}. Skipping sync.")

    def force_free(self, *handles: Handle):
        """Force free handles immediately (outside of allocate context). Adjust size counters and wake waiters."""
        with self._lock_metadata:
            freed_bytes = 0
            for handle in handles:
                tensor = self._allocated_tensors.pop(handle, None)
                size_bytes = self._handle_size_bytes.pop(handle, 0)
                if tensor is not None:
                    freed_bytes += size_bytes
                    logger.info(f"OFFLOAD: Force free KV tensor handle={handle}, bytes={size_bytes}")
                if handle in self._underlying:
                    self._underlying.pop(handle, None)
                    self._layout_meta.pop(handle, None)
                    self._cpu_stage.pop(handle, None)
                    self._dtype_by_handle.pop(handle, None)
            if freed_bytes:
                self.current_size_bytes = max(0, self.current_size_bytes - freed_bytes)
                self._memory_freed_event.set()


class AllocationFailed(Exception):
    pass

class KVCacheMetadata:
    device: Any               # 存在哪个设备上
    offloaded: bool = False             # 是否已 offload 到 CPU
    # TODO: add more device info


@dataclasses.dataclass(frozen=True)
class AdaptedKVCache:
    kvs: Sequence[torch.Tensor]
    device: KVCacheMetadata


