"""
Tree management operations: traversal, status updates, branch statistics.
"""
from typing import List, Dict, Optional
from data_models import TaxonomyNode, NodeStatus, BranchStats


class TreeManager:
    """Manages tree operations and branch statistics."""
    
    def __init__(self, root: TaxonomyNode):
        self.root = root
        self._branch_stats: Dict[str, BranchStats] = {}
        self._initialize_branch_stats()
    
    def _initialize_branch_stats(self):
        """Initialize branch statistics for all top-level branches."""
        for child in self.root.children:
            branch_id = self._get_branch_id(child)
            self._branch_stats[branch_id] = BranchStats(
                root_child=child,
                total_visits=0,
                total_nodes=0,
                killed_count=0,
                verified_relevant_count=0
            )
    
    def _get_branch_id(self, node: TaxonomyNode) -> str:
        """Get the branch ID for a node (top-level child ID)."""
        current = node
        while current.parent is not None and current.parent != self.root:
            current = current.parent
        return current.id
    
    def get_all_nodes(self) -> List[TaxonomyNode]:
        """
        Get ALL nodes in the tree (status doesn't matter).
        """
        all_nodes = []
        
        def traverse(node: TaxonomyNode):
            if node.id != "root":
                all_nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(self.root)
        return all_nodes

    def get_alive_nodes(self, subtree_root: Optional[TaxonomyNode] = None) -> List[TaxonomyNode]:
        """
        Get all alive nodes in the tree (or subtree).
        
        Args:
            subtree_root: If provided, only traverse this subtree. Otherwise, traverse entire tree.
        
        Returns:
            List of all alive nodes (not just leaves).
        """
        if subtree_root is None:
            subtree_root = self.root
        
        alive_nodes = []
        
        def traverse(node: TaxonomyNode):
            if node.is_alive() and node.id != "root":
                alive_nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(subtree_root)
        return alive_nodes
    
    def kill_subtree(self, node: TaxonomyNode):
        """
        Mark node and all its descendants as KILLED.
        
        Note: Does NOT kill ancestors - only propagates downward.
        """
        def kill_recursive(n: TaxonomyNode):
            n.status = NodeStatus.KILLED
            for child in n.children:
                kill_recursive(child)
        
        kill_recursive(node)
        self._update_branch_stats()
    
    def mark_human_verified(self, node: TaxonomyNode, label: bool):
        """
        Mark a node as verified by human.
        
        Args:
            node: Node to mark
            label: True if relevant, False if irrelevant
        """
        if label:
            node.status = NodeStatus.HUMAN_VERIFIED_RELEVANT
            node.human_label = True
        else:
            # If irrelevant, kill the subtree
            self.kill_subtree(node)
            node.human_label = False
        
        self._update_branch_stats()
    
    def get_branch_stats(self, branch_id: Optional[str] = None) -> Dict[str, BranchStats]:
        """
        Get branch statistics.
        
        Args:
            branch_id: If provided, return stats for this branch only
        
        Returns:
            Dictionary mapping branch_id to BranchStats
        """
        if branch_id:
            return {branch_id: self._branch_stats.get(branch_id)}
        return self._branch_stats.copy()
    
    def get_branch_for_node(self, node: TaxonomyNode) -> Optional[BranchStats]:
        """Get branch statistics for the branch containing this node."""
        branch_id = self._get_branch_id(node)
        return self._branch_stats.get(branch_id)
    
    def increment_visit(self, node: TaxonomyNode):
        """Increment visit count for node and update branch statistics."""
        node.visit_count += 1
        self._update_branch_stats()
    
    def _update_branch_stats(self):
        """Recalculate branch statistics."""
        # Reset all stats
        for stats in self._branch_stats.values():
            stats.total_visits = 0
            stats.total_nodes = 0
            stats.killed_count = 0
            stats.verified_relevant_count = 0
        
        # Recount
        def traverse(node: TaxonomyNode, branch_stats: BranchStats):
            branch_stats.total_nodes += 1
            branch_stats.total_visits += node.visit_count
            
            if node.status == NodeStatus.KILLED:
                branch_stats.killed_count += 1
            elif node.status == NodeStatus.HUMAN_VERIFIED_RELEVANT:
                branch_stats.verified_relevant_count += 1
            
            for child in node.children:
                traverse(child, branch_stats)
        
        for branch_id, stats in self._branch_stats.items():
            # stats.root_child is the top-level node for this branch
            traverse(stats.root_child, stats)
    
    def get_total_visits(self) -> int:
        """Get total number of visits across all nodes."""
        return sum(stats.total_visits for stats in self._branch_stats.values())

    def get_node(self, node_id: str) -> Optional[TaxonomyNode]:
        """
        Find a node by ID using BFS.
        
        Args:
            node_id: ID of the node to find
            
        Returns:
            The node object if found, else None
        """
        if self.root.id == node_id:
            return self.root
            
        queue = [self.root]
        while queue:
            curr = queue.pop(0)
            if curr.id == node_id:
                return curr
            queue.extend(curr.children)
        return None
