import uuid
import math
from typing import List, Optional, Dict, Tuple, Any
import torch

class TreeNode:
    """推测树的基本节点"""
    
    def __init__(self, token_id: int, probability: float = 1.0, depth: int = 0):
        self.token_id = token_id
        self.probability = probability
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
        self.node_id = str(uuid.uuid4())
        self.position_in_sequence = -1
        
    def add_child(self, token_id: int, probability: float) -> 'TreeNode':
        """添加子节点"""
        child = TreeNode(token_id, probability, self.depth + 1)
        child.parent = self
        self.children.append(child)
        return child
    
    def add_children(self, candidates: List[Tuple[int, float]]) -> List['TreeNode']:
        """批量添加子节点"""
        children = []
        for token_id, prob in candidates:
            child = self.add_child(token_id, prob)
            children.append(child)
        return children
    
    def get_path_from_root(self) -> List[int]:
        """获取从根节点到当前节点的token路径"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.token_id)
            current = current.parent
        return list(reversed(path))
    
    def get_path_nodes_from_root(self) -> List['TreeNode']:
        """获取从根节点到当前节点的节点路径"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0
    
    def get_all_leaf_paths(self) -> List[List[int]]:
        """获取所有从根到叶子的路径"""
        if self.is_leaf():
            return [self.get_path_from_root()]
        
        all_paths = []
        for child in self.children:
            child_paths = child.get_all_leaf_paths()
            all_paths.extend(child_paths)
        return all_paths
    
    def get_all_leaf_node_paths(self) -> List[List['TreeNode']]:
        """获取所有从根到叶子的节点路径"""
        if self.is_leaf():
            return [self.get_path_nodes_from_root()]
        
        all_paths = []
        for child in self.children:
            child_paths = child.get_all_leaf_node_paths()
            all_paths.extend(child_paths)
        return all_paths
    
    def __str__(self):
        return f"TreeNode(token={self.token_id}, prob={self.probability:.3f}, depth={self.depth})"


class SpeculativeTree:
    """单个请求的推测树"""
    
    def __init__(self, root_token: int, request_id: str):
        self.root = TreeNode(root_token, 1.0, 0)
        self.request_id = request_id
        self.max_depth = 0
        self.total_nodes = 1
        
    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """获取指定深度的所有节点"""
        if depth == 0:
            return [self.root]
        
        nodes = []
        def traverse(node, current_depth):
            if current_depth == depth:
                nodes.append(node)
            elif current_depth < depth:
                for child in node.children:
                    traverse(child, current_depth + 1)
        
        traverse(self.root, 0)
        return nodes
    
    def add_layer(self, parent_nodes: List[TreeNode], candidates_per_node: List[List[Tuple[int, float]]]):
        """为指定的父节点添加一层候选"""
        if len(parent_nodes) != len(candidates_per_node):
            raise ValueError("父节点数量与候选数量不匹配")
        
        new_nodes = []
        for parent, candidates in zip(parent_nodes, candidates_per_node):
            children = parent.add_children(candidates)
            new_nodes.extend(children)
            
        if new_nodes:
            self.max_depth = max(self.max_depth, max(node.depth for node in new_nodes))
            self.total_nodes += len(new_nodes)
        
        return new_nodes
    
    def get_all_paths(self) -> List[List[int]]:
        """获取所有可能的路径"""
        return self.root.get_all_leaf_paths()


    def linearize_tree_with_positions(tree: SpeculativeTree) -> Tuple[List[TreeNode], List[int]]:
        """
        DFS线性化: 记录父位置
        """
        linearized_nodes = []
        parent_indices = []
        position_map = {}
        
        def dfs_with_positions(node):
            if node.parent is not None:  # 跳过root
                pos = len(linearized_nodes)
                position_map[node] = pos
                node.position_in_sequence = pos
                linearized_nodes.append(node)
                
                parent_pos = position_map.get(node.parent, -1)
                parent_indices.append(parent_pos)
            
            for child in node.children:
                dfs_with_positions(child)
        
        dfs_with_positions(tree.root)
        return linearized_nodes, parent_indices


    def build_ancestor_matrix_optimized(parent_indices: List[int], device: torch.device) -> torch.Tensor:
        """
        修复问题1: 使用scatter_和bit-jump优化祖先矩阵构建
        """
        n = len(parent_indices)
        if n == 0:
            return torch.empty(0, 0, dtype=torch.bool, device=device)
        
        # 使用scatter_一次性构建直接父子关系
        A = torch.zeros(n, n, dtype=torch.bool, device=device)
        
        rows = torch.arange(n, device=device)
        cols = torch.as_tensor(parent_indices, device=device)
        mask = cols >= 0  # 有效的父节点
        
        if mask.any():
            A[rows[mask], cols[mask]] = True
        
        # 使用bit-jump优化传递闭包
        ancestor_matrix = A.clone()
        k = 1
        
        while k < n:
            # 计算k步可达关系
            power_A = torch.matmul(ancestor_matrix, A)
            new_reachable = ancestor_matrix | power_A
            
            if torch.equal(new_reachable, ancestor_matrix):
                break
                
            ancestor_matrix = new_reachable
            k *= 2
        
        return ancestor_matrix


    def build_tree_attention_mask_batched(
        prefix_len: int,
        max_tree_size: int,
        all_parent_indices: List[List[int]],
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        修复问题2: 正确处理batch维度的attention mask
        """
        total_len = prefix_len + max_tree_size
        
        # 为每个batch创建mask
        mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
        mask = mask.expand(batch_size, -1, -1).clone()  # [batch_size, total_len, total_len]
        
        for batch_idx, parent_indices in enumerate(all_parent_indices):
            tree_len = len(parent_indices)
            if tree_len == 0:
                continue
            
            # 构建祖先矩阵
            ancestor_matrix = build_ancestor_matrix_optimized(parent_indices, device)
            
            # 批量更新树部分的mask
            tree_start = prefix_len
            
            # 创建索引张量用于批量更新
            tree_indices = torch.arange(tree_len, device=device)
            tree_positions_i = tree_start + tree_indices.unsqueeze(1)  # [tree_len, 1]
            tree_positions_j = tree_start + tree_indices.unsqueeze(0)  # [1, tree_len]
            
            # 只有祖先关系或自己的位置才能看到
            valid_attention = ancestor_matrix | torch.eye(tree_len, dtype=torch.bool, device=device)
            
            # 批量更新：无效的attention位置设为False
            invalid_mask = ~valid_attention
            if invalid_mask.any():
                invalid_i = tree_positions_i.expand(tree_len, tree_len)[invalid_mask]
                invalid_j = tree_positions_j.expand(tree_len, tree_len)[invalid_mask]
                mask[batch_idx, invalid_i, invalid_j] = False
        
        return mask


    def prepare_tree_attention_batch(
        trees: List[SpeculativeTree], 
        prefix_tokens: torch.Tensor,
        device: torch.device,
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[TreeNode]]]]:
        """
        批量处理多个树的线性化和attention mask构建
        """
        batch_size = len(trees)
        
        if not trees or all(tree.total_nodes <= 1 for tree in trees):
            return prefix_tokens, None, [[] for _ in trees]
        
        max_tree_size = max(tree.total_nodes - 1 for tree in trees if tree.total_nodes > 1)
        
        # 收集线性化结果
        batch_linearized = []
        batch_node_paths = []
        all_parent_indices = []
        
        for tree in trees:
            linearized_nodes, parent_indices = linearize_tree_with_positions(tree)
            batch_linearized.append([node.token_id for node in linearized_nodes])
            batch_node_paths.append(tree.root.get_all_leaf_node_paths())
            all_parent_indices.append(parent_indices)
        
        # 填充到相同长度
        padded_tokens = []
        for tokens in batch_linearized:
            padded = tokens + [pad_token_id] * (max_tree_size - len(tokens))
            padded_tokens.append(padded)
        
        tree_tokens = torch.tensor(padded_tokens, device=device)
        full_sequence = torch.cat([prefix_tokens, tree_tokens], dim=-1)
        
        # 构建批量attention mask
        attention_mask = build_tree_attention_mask_batched(
            prefix_tokens.shape[1], max_tree_size, all_parent_indices, batch_size, device
        )
        
        return full_sequence, attention_mask, batch_node_paths