"""
Samples nodes for human grading using UCB-aligned strategy.
"""
from typing import List
from data_models import TaxonomyNode
from tree_manager import TreeManager
from expansion_selector import ExpansionSelector


class HumanGradingSampler:
    """
    Samples nodes for human review using UCB-aligned strategy:
    - 50% from top UCB scores (validate expansion targets)
    - 50% from low rubric scores (kill bad subtrees early)
    """
    
    def __init__(
        self,
        tree_manager: TreeManager,
        expansion_selector: ExpansionSelector
    ):
        self.tree_manager = tree_manager
        self.expansion_selector = expansion_selector
    
    def select_for_human_grading(self, k: int, excluded_ids: List[str] = None) -> List[TaxonomyNode]:
        """
        Select K nodes for human grading.
        
        Args:
            k: Number of nodes to select
            excluded_ids: Optional list of node IDs to exclude (e.g. already queued)
        
        Returns:
            List of nodes to present to human
        """
        alive_nodes = self.tree_manager.get_alive_nodes()
        excluded_set = set(excluded_ids) if excluded_ids else set()
        
        # Filter out already-labeled nodes, root, and excluded IDs
        unlabeled_nodes = [
            n for n in alive_nodes 
            if n.human_label is None and n.depth > 0 and n.id not in excluded_set
        ]
        
        if len(unlabeled_nodes) < k:
            # Not enough unlabeled nodes
            return unlabeled_nodes
        
        # Compute UCB scores for all nodes
        ucb_scores = []
        rubric_scores = []
        
        for node in unlabeled_nodes:
            ucb_score = self.expansion_selector.compute_ucb_score(node)
            ucb_scores.append((node, ucb_score))
            rubric_scores.append((node, node.rubric_score))
        
        # 50% from top UCB (validate expansion targets)
        k_top_ucb = k // 2
        top_ucb_nodes = self._top_k(ucb_scores, k_top_ucb)
        
        # 50% from low rubric (kill bad subtrees early)
        k_low_rubric = k - k_top_ucb
        low_rubric_nodes = self._bottom_k(rubric_scores, k_low_rubric)
        
        # Combine and dedupe by ID
        combined = top_ucb_nodes + low_rubric_nodes
        seen_ids = set()
        selected = []
        for node in combined:
            if node.id not in seen_ids:
                selected.append(node)
                seen_ids.add(node.id)
        
        # If we have fewer than k due to deduplication, fill with remaining high UCB
        if len(selected) < k:
            remaining = [n for n, _ in ucb_scores if n not in selected]
            remaining.sort(key=lambda n: n.ucb_score, reverse=True)
            selected.extend(remaining[:k - len(selected)])
        
        return selected[:k]
    
    def _top_k(self, scored_items: List[tuple], k: int) -> List[TaxonomyNode]:
        """Get top K items by score."""
        sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_items[:k]]
    
    def _bottom_k(self, scored_items: List[tuple], k: int) -> List[TaxonomyNode]:
        """Get bottom K items by score."""
        sorted_items = sorted(scored_items, key=lambda x: x[1])
        return [node for node, _ in sorted_items[:k]]
