import uuid
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class TreeNode:
    """推测树的基本节点"""
    
    def __init__(self, token_id: int, probability: float = 1.0, depth: int = 0):
        self.token_id = token_id
        self.probability = probability
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
        self.node_id = str(uuid.uuid4())  # 唯一标识符
        
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
    
    def get_path_with_probs(self) -> List[Tuple[int, float]]:
        """获取路径及对应概率"""
        path = []
        current = self
        while current.parent is not None:
            path.append((current.token_id, current.probability))
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
    
    def get_subtree_size(self) -> int:
        """获取子树大小（包括自身）"""
        size = 1
        for child in self.children:
            size += child.get_subtree_size()
        return size
    
    def __str__(self):
        return f"TreeNode(token={self.token_id}, prob={self.probability:.3f}, depth={self.depth})"
    
    def __repr__(self):
        return self.__str__()
    
    
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
    
    def get_leaf_nodes(self) -> List[TreeNode]:
        """获取所有叶子节点"""
        leaves = []
        def traverse(node):
            if node.is_leaf():
                leaves.append(node)
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(self.root)
        return leaves
    
    def add_layer(self, parent_nodes: List[TreeNode], candidates_per_node: List[List[Tuple[int, float]]]):
        """为指定的父节点添加一层候选"""
        if len(parent_nodes) != len(candidates_per_node):
            raise ValueError("父节点数量与候选数量不匹配")
        
        new_nodes = []
        for parent, candidates in zip(parent_nodes, candidates_per_node):
            children = parent.add_children(candidates)
            new_nodes.extend(children)
            
        # 更新统计信息
        if new_nodes:
            self.max_depth = max(self.max_depth, max(node.depth for node in new_nodes))
            self.total_nodes += len(new_nodes)
        
        return new_nodes
    
    def get_all_paths(self) -> List[List[int]]:
        """获取所有可能的路径"""
        return self.root.get_all_leaf_paths()
    
    def get_all_paths_with_scores(self) -> List[Tuple[List[int], float]]:
        """获取所有路径及其累积概率"""
        paths_with_scores = []
        
        def traverse(node, current_path, current_score):
            current_path = current_path + [node.token_id] if node.parent else current_path
            
            if node.is_leaf():
                if current_path:  # 排除根节点
                    paths_with_scores.append((current_path, current_score))
            else:
                for child in node.children:
                    traverse(child, current_path, current_score * child.probability)
        
        traverse(self.root, [], 1.0)
        return paths_with_scores
    
    def get_best_path(self) -> Tuple[List[int], float]:
        """获取概率最高的路径"""
        paths_with_scores = self.get_all_paths_with_scores()
        if not paths_with_scores:
            return [], 0.0
        
        return max(paths_with_scores, key=lambda x: x[1])
    
    def get_top_k_paths(self, k: int) -> List[Tuple[List[int], float]]:
        """获取概率最高的k条路径"""
        paths_with_scores = self.get_all_paths_with_scores()
        sorted_paths = sorted(paths_with_scores, key=lambda x: x[1], reverse=True)
        return sorted_paths[:k]
    
    def prune_low_probability_branches(self, threshold: float):
        """剪枝低概率分支"""
        def should_prune(node):
            return node.probability < threshold
        
        def prune_recursive(node):
            # 从后往前遍历，避免索引问题
            for i in range(len(node.children) - 1, -1, -1):
                child = node.children[i]
                if should_prune(child):
                    node.children.pop(i)
                    self.total_nodes -= child.get_subtree_size()
                else:
                    prune_recursive(child)
        
        prune_recursive(self.root)
    
    def print_tree(self, max_depth: Optional[int] = None):
        """打印树结构"""
        def print_node(node, prefix="", is_last=True):
            if max_depth is not None and node.depth > max_depth:
                return
                
            print(f"{prefix}{'└── ' if is_last else '├── '}{node}")
            
            if node.children:
                for i, child in enumerate(node.children):
                    is_child_last = i == len(node.children) - 1
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    print_node(child, child_prefix, is_child_last)
        
        print(f"SpeculativeTree for request {self.request_id}:")
        print_node(self.root)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        def node_to_dict(node):
            return {
                'token_id': node.token_id,
                'probability': node.probability,
                'depth': node.depth,
                'children': [node_to_dict(child) for child in node.children]
            }
        
        return {
            'request_id': self.request_id,
            'max_depth': self.max_depth,
            'total_nodes': self.total_nodes,
            'root': node_to_dict(self.root)
        }