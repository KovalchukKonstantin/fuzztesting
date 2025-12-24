"""
Two-Level UCB expansion selector with New Branch arm.
"""
import math
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from data_models import TaxonomyNode, BranchStats, Project
from tree_manager import TreeManager


NEW_BRANCH_ARM = "NEW_BRANCH"  # Special identifier for creating new branches


class ExpansionSelector:
    """
    Selects nodes for expansion using Two-Level UCB.
    
    Level 1: Select branch (or NEW_BRANCH)
    Level 2: Select node within branch
    """
    
    def __init__(
        self,
        tree_manager: TreeManager,
        project: Project,
        c1: float = 1.0,  # Branch exploration weight
        c2: float = 1.4,  # Node exploration weight
        c_new: float = 1.5,  # New branch exploration weight
        depth_factor: float = 0.2  # Depth penalty factor
    ):
        self.tree_manager = tree_manager
        self.project = project
        self.c1 = c1
        self.c2 = c2
        self.c_new = c_new
        self.depth_factor = depth_factor
    
    def select_for_expansion(self, m: int) -> List[TaxonomyNode]:
        """
        Select M nodes for expansion.
        
        Args:
            m: Number of nodes to select
        
        Returns:
            List of nodes to expand (or empty list if NEW_BRANCH selected)
        """
        selected_nodes = []
        
        for _ in range(m):
            # Level 1: Select branch or NEW_BRANCH
            selected_branch = self._select_branch()
            
            if selected_branch == NEW_BRANCH_ARM:
                # Create new branch - handled by caller
                # Return empty list to signal new branch creation
                return []
            else:
                # Level 2: Select node within branch
                # Exclude already selected nodes to ensure diversity in batch
                exclude_ids = {n.id for n in selected_nodes}
                node = self._select_node_in_branch(selected_branch, exclude_node_ids=exclude_ids)
                if node:
                    selected_nodes.append(node)
        
        return selected_nodes
    
    def _select_branch(self) -> TaxonomyNode | str:
        """
        Level 1: Select which branch to work on (or NEW_BRANCH).
        
        Returns:
            Branch node or NEW_BRANCH constant
        """
        branches = self.tree_manager.root.children
        branch_scores = []
        
        for branch in branches:
            if not branch.is_alive():
                continue
            
            # Calculate branch UCB score
            branch_stats = self.tree_manager.get_branch_for_node(branch)
            if branch_stats is None:
                continue
            
            avg_rubric = self._avg_rubric_score_in_branch(branch)
            n_total = self.tree_manager.get_total_visits()
            
            branch_exploration = self.c1 * math.sqrt(
                math.log(max(1, n_total)) / max(1, branch_stats.total_visits)
            )
            
            branch_score = avg_rubric + branch_exploration
            branch_scores.append((branch, branch_score))
        
        # Add NEW_BRANCH arm
        n_total = self.tree_manager.get_total_visits()
        new_branch_score = self.c_new * math.sqrt(
            math.log(max(1, n_total)) / max(1, self.project.new_branch_count + 1)
        )
        branch_scores.append((NEW_BRANCH_ARM, new_branch_score))
        
        # Select highest scoring
        if not branch_scores:
            return NEW_BRANCH_ARM
        
        best_branch, _ = max(branch_scores, key=lambda x: x[1])
        return best_branch
    
    def _select_node_in_branch(
        self, 
        branch: TaxonomyNode, 
        exclude_node_ids: Optional[set] = None
    ) -> Optional[TaxonomyNode]:
        """
        Level 2: Select node within branch using UCB with depth penalty.
        
        Args:
            branch: Branch to select from
            exclude_node_ids: Set of node IDs to exclude from selection
        
        Returns:
            Selected node or None if no alive nodes in branch
        """
        nodes = self.tree_manager.get_alive_nodes(branch)
        if exclude_node_ids:
            nodes = [n for n in nodes if n.id not in exclude_node_ids]
            
        if not nodes:
            return None
        
        branch_stats = self.tree_manager.get_branch_for_node(branch)
        if branch_stats is None:
            return None
        
        # Calculate UCB for each node
        node_scores = []
        for node in nodes:
            depth = node.depth
            depth_discount = 1 / (1 + depth * self.depth_factor)
            
            node_exploration = self.c2 * math.sqrt(
                math.log(max(1, branch_stats.total_visits)) / max(1, node.visit_count)
            ) * depth_discount
            
            ucb_score = node.rubric_score + node_exploration
            node.ucb_score = ucb_score
            node_scores.append((node, ucb_score))
        
        # Select highest scoring
        best_node, _ = max(node_scores, key=lambda x: x[1])
        return best_node
    
    def _avg_rubric_score_in_branch(self, branch: TaxonomyNode) -> float:
        """Calculate average rubric score for alive nodes in branch."""
        nodes = self.tree_manager.get_alive_nodes(branch)
        if not nodes:
            return 0.0
        
        total_score = sum(node.rubric_score for node in nodes)
        return total_score / len(nodes)
    
    def compute_ucb_score(self, node: TaxonomyNode) -> float:
        """
        Compute UCB score for a node (for external use, e.g., human grading sampler).
        
        Args:
            node: Node to score
        
        Returns:
            UCB score
        """
        branch_stats = self.tree_manager.get_branch_for_node(node)
        if branch_stats is None:
            return node.rubric_score
        
        depth = node.depth
        depth_discount = 1 / (1 + depth * self.depth_factor)
        
        n_total = self.tree_manager.get_total_visits()
        node_exploration = self.c2 * math.sqrt(
            math.log(max(1, n_total)) / max(1, node.visit_count)
        ) * depth_discount
        
        branch_exploration = self.c1 * math.sqrt(
            math.log(max(1, n_total)) / max(1, branch_stats.total_visits)
        )
        
        ucb_score = node.rubric_score + branch_exploration + node_exploration
        node.ucb_score = ucb_score
        return ucb_score
