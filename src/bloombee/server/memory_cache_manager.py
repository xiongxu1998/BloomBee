from dataclasses import dataclass
import contextlib
import torch
import os
from typing import Optional, Tuple, AsyncContextManager, Sequence
from bloombee.server.memory_cache import MemoryCache, AdaptedKVCache, KVCacheMetadata
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy

from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from bloombee.data_structures import Handle
from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.misc import get_size_in_bytes

from transformers import PretrainedConfig


class KVCacheManager:
    def __init__(self, cache_max_size: int, 
                 max_alloc_timeout: int, 
                 policy: Policy, 
                 env: ExecutionEnv,
                 block_config: PretrainedConfig):
        # 初始化为二维数组结构
        self.env = env
        self.runtime_pid = os.getpid()
        self.cache = MemoryCache(cache_max_size, max_alloc_timeout, policy, block_config, self.get_cache_device(policy))
        self.offloading_policy = policy
        self.attention_compute = (self.env.cpu if policy.cpu_cache_compute
                                  else self.env.gpu)
        self.block_config = block_config
        
        
    def get_cache_device(self, policy):
        if policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        return device


    def clear(self):
        # for b in range(self.max_batch_size):
        #     for l in range(self.num_layers):
        #         self.cache[b][l] = None
        pass
    
    @contextlib.asynccontextmanager
    async def allocate_cache(
        self, *descriptors: TensorDescriptor, timeout: float
    ) -> AsyncContextManager[Sequence[Handle]]:
        """
        Create a handle that is associated with buffers on unique device. If cache full, raises AllocationFailed.

        :param descriptors: one or more tensors tensor of this size, dtype, etc
        :param timeout: optional maximum time to wait for cache allocation; None (default) means no time limit

        :note: if descriptors reside on different devices, it is expected that they are approximately balanced across devices;
          if not, it will count maximum tensor allocation across devices for the purposes of size limit

        :note: This function should be called by connection handlers, it can be called concurrently from multiple processes.
        Furthermore, it can be called concurrently with at most one use_cache call in runtime.
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)
        max_alloc_size = self.get_allocation_size(*descriptors)

        gib = 1024**3
        cur_size, max_size = self.current_size_bytes, self.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / max_size * 100:.1f}%)"
        )

        alloc_task = asyncio.create_task(self.cache._schedule_alloc(max_alloc_size, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self.cache._free(max_alloc_size, alloc_task)
            
            
    @staticmethod
    def get_allocation_size(*descriptors: TensorDescriptor) -> int:
        """Return the memory size (bytes) to be allocated on a device. If there are many devices, return maximum"""
        alloc_size_by_device = {}
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] = alloc_size_by_device.get(descr.device, 0) + tensor_size
        return max(alloc_size_by_device.values())
        
    
    def add_cache(self, kvs: AdaptedKVCache, start_position: int):
        pass
                
    def update_cache(
        self, new_kvs: AdaptedKVCache, start_position: int
    ):
        pass
    
    def bytes_left(self) -> int:
        return self.cache.bytes_left
    
    def select_cache(self, kv_cache_position_ids: Optional[torch.Tensor] = None):
        pass
    
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        with self.cache.use_cache(*handles) as cache_tensors:
            yield cache_tensors
        
        
