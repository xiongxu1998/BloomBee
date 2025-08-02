from dataclasses import dataclass
import torch
from typing import Optional, Tuple
from bloombee.server.memory_cache import MemoryCache
from bloombee.data_structures import KVCache, KVCacheMetadata
from bloombee.flexgen_utils.policy import Policy


class KVCacheManager:
    def __init__(self, cache_size: int, max_alloc_timeout: int, policy: Policy):
        # 初始化为二维数组结构
        self.cache = MemoryCache(cache_size, max_alloc_timeout)
        self.offloading_policy = policy

    def clear(self):
        # for b in range(self.max_batch_size):
        #     for l in range(self.num_layers):
        #         self.cache[b][l] = None
        pass
    
    def add_cache(self, kvs: KVCache, start_position: int):
        pass
                
    def update_cache(
        self, new_kvs: KVCache, start_position: int
    ):
        pass
    
    def bytes_left(self) -> int:
        return self.cache.bytes_left()
    
    def select_cache(self, kv_cache_position_ids: Optional[torch.Tensor] = None):
        pass
    
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        with self.memory_cache.use_cache(*handles) as cache_tensors:
            yield cache_tensors
        
        
