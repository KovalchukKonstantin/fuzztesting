"""
Main orchestrator that coordinates all components and runs the iteration loop.
"""
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from storage.adapter import AsyncStorageAdapter

from data_models import Project, TaxonomyNode, LabeledSample, Rubric, PrincipleScore
from tree_manager import TreeManager
from expansion_selector import ExpansionSelector, NEW_BRANCH_ARM
from human_grading_sampler import HumanGradingSampler
from rubric_scorer import RubricScorer
from rubric_deriver import RubricDeriver
from taxonomy_generator import TaxonomyGenerator
import logging
import asyncio

logger = logging.getLogger(__name__)


class TestSuiteOrchestrator:
    """
    Main orchestrator for the test suite generation system.
    
    Coordinates all components and manages the iteration loop.
    """
    
    def __init__(
        self,
        project: Project,
        taxonomy_generator: TaxonomyGenerator,
        rubric_scorer: RubricScorer,
        rubric_deriver: RubricDeriver,
        expansion_selector: ExpansionSelector,
        human_grading_sampler: HumanGradingSampler,
        tree_manager: TreeManager,
        storage: Optional['AsyncStorageAdapter'] = None
    ):
        """
        Initialize orchestrator with all components.
        
        Args:
            project: Project instance
            taxonomy_generator: Taxonomy generator
            rubric_scorer: Rubric scorer
            rubric_deriver: Rubric deriver
            expansion_selector: Expansion selector
            human_grading_sampler: Human grading sampler
            tree_manager: Tree manager
            storage: Optional async storage adapter for persistence
        """
        self.project = project
        self.taxonomy_generator = taxonomy_generator
        self.rubric_scorer = rubric_scorer
        self.rubric_deriver = rubric_deriver
        self.expansion_selector = expansion_selector
        self.human_grading_sampler = human_grading_sampler
        self.tree_manager = tree_manager
        self.storage = storage
        # Initialize iteration count from existing rubrics to resume correctly
        self._iteration_count = len(project.rubrics) if project.rubrics else 0
    
    async def initialize(self, num_initial_topics: int = 5):
        """
        Initialize the taxonomy tree.
        
        Persists to storage if configured.
        
        Args:
            num_initial_topics: Number of initial top-level topics
        """
        root = await self.taxonomy_generator.generate_initial_tree(
            self.project.product_description,
            num_initial_topics
        )
        logger.info(f"Generated initial taxonomy tree with {len(root.children)} topics")
        self.project.taxonomy_root = root
        
        # Reinitialize tree manager with new root
        self.tree_manager = TreeManager(root)
        self.expansion_selector.tree_manager = self.tree_manager
        self.human_grading_sampler.tree_manager = self.tree_manager
        
        # Persist if storage is configured
        if self.storage:
            await self.save_state()
    
    def step_human_grading(self, k: int) -> List[TaxonomyNode]:
        """
        Step 1: Select nodes for human grading.
        
        Args:
            k: Number of nodes to select
        
        Returns:
            List of nodes to present to human
        """
        logger.info(f"Stepping human grading: selecting {k} nodes")
        return self.human_grading_sampler.select_for_human_grading(k)
    
    def step_update_from_feedback(
        self,
        labeled_nodes: List[tuple[TaxonomyNode, bool]]
    ):
        """
        Step 2-3: Update tree based on human feedback.
        
        Args:
            labeled_nodes: List of (node, label) tuples where label is True (relevant) or False (irrelevant)
        """
        iteration = self._iteration_count
        logger.info(f"Processing feedback for {len(labeled_nodes)} nodes (Iteration {iteration})")
        
        for node, label in labeled_nodes:
            # Mark in tree
            self.tree_manager.mark_human_verified(node, label)
            
            # Record in project
            sample = LabeledSample(
                node=node,
                label=label,
                iteration=iteration
            )
            self.project.human_labeled_samples.append(sample)
    
    async def step_derive_rubrics(self) -> Tuple[Optional[Rubric], int, int]:
        """
        Step 4: Derive/update rubrics from human feedback.
        
        Returns:
            Tuple: (Updated rubric or None, new_count, merged_count)
        """
        if not self.project.human_labeled_samples:
            return None, 0, 0
        
        return await self.rubric_deriver.derive_rubrics(self.project)
    
    async def step_rescore_all(self) -> dict:
        """
        Step 5: Re-score all nodes (including killed) with updated rubrics.
        
        Only scores nodes against new principles (uses cached scores for existing ones).
        
        Returns:
            Dictionary mapping node_id to (score, principle_scores) for caching
        """
        rubric = self.project.get_current_rubric()
        if not rubric:
            return {}
        
        # Rescore EVERYTHING (including killed nodes) so metrics are accurate
        all_nodes = self.tree_manager.get_all_nodes()
        
        # Get cached scores from storage if available
        cached_scores = {}
        if self.storage and rubric.principles:
            # Get all principle IDs
            principle_ids = [p.id for p in rubric.principles if p.id]
            if principle_ids:
                node_ids = [node.id for node in all_nodes]
                async with self.storage.database.get_session() as session:
                    from storage.repositories import PrincipleScoreRepository
                    repo = PrincipleScoreRepository(session)
                    cached_scores = await repo.get_cached_scores(node_ids, principle_ids)
        
        # Score nodes (only new principles will be scored)
        # Note: function name says 'alive' but it takes any list of nodes
        results = await self.rubric_scorer.score_all_alive(all_nodes, rubric, self.project.product_description, cached_scores)
        
        # Recompute UCB scores (only matters for alive nodes, but harmless for killed)
        for node in all_nodes:
            self.expansion_selector.compute_ucb_score(node)
        
        return results
    
    async def step_expand(self, m: int, num_children_per_node: int = 3) -> List[TaxonomyNode]:
        """
        Step 6: Expand taxonomy by selecting and expanding nodes.
        
        Args:
            m: Number of nodes to expand
            num_children_per_node: Number of children to generate per node
        
        Returns:
            List of newly created nodes (or empty if NEW_BRANCH was selected)
        """
        selected_nodes = self.expansion_selector.select_for_expansion(m)
        
        # Check if NEW_BRANCH was selected
        if not selected_nodes:
            # Create new branch
            new_branch = await self.taxonomy_generator.create_new_branch(
                self.project,
                current_iteration=self._iteration_count
            )
            logger.info(f"New branch created: {new_branch.content}")
            # Reinitialize tree manager to pick up new branch
            self.tree_manager = TreeManager(self.project.taxonomy_root)
            self.expansion_selector.tree_manager = self.tree_manager
            self.human_grading_sampler.tree_manager = self.tree_manager
            return [new_branch]
        
        # Expand selected nodes
        rubric = self.project.get_current_rubric()
        
        async def process_expansion(node: TaxonomyNode) -> Tuple[List[TaxonomyNode], Dict[str, List[PrincipleScore]]]:
            # Expand single node
            children = await self.taxonomy_generator.expand_node(
                node,
                rubric,
                self.project.product_description,
                num_children_per_node,
                current_iteration=self._iteration_count
            )
            
            # Update visit counts
            self.tree_manager.increment_visit(node)
            
            # Score new children
            node_principle_scores = {}
            if rubric:
                for child in children:
                    score, p_scores = await self.rubric_scorer.score_node(child, rubric, self.project.product_description)
                    child.rubric_score = score
                    node_principle_scores[child.id] = p_scores
            
            return children, node_principle_scores

        # Create tasks for all selected nodes
        tasks = [process_expansion(node) for node in selected_nodes]
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        # Flatten results and collect scores
        new_nodes = []
        all_new_principle_scores = {}
        for children, p_scores_map in results:
            new_nodes.extend(children)
            all_new_principle_scores.update(p_scores_map)
            
        # Persist new nodes and their principle scores if storage available
        if self.storage:
            await self.persist_after_expansion(new_nodes)
            await self.flush_storage()  # Ensure nodes are saved before scores
            if all_new_principle_scores:
                iteration = len(self.project.rubrics)  # Current rubric iteration
                await self.persist_principle_scores(all_new_principle_scores, iteration)
        
        self.project.total_expansions += 1
        
        logger.info(f"Expanded {len(selected_nodes)} nodes into {len(new_nodes)} new children")
        
        return new_nodes
    
    def run_iteration(
        self,
        k: int = 10,  # Nodes for human grading
        m: int = 5,   # Nodes to expand
        num_children_per_node: int = 3
    ) -> dict:
        """
        Run one full iteration of the main loop.
        
        Args:
            k: Number of nodes to select for human grading
            m: Number of nodes to expand
            num_children_per_node: Number of children per expanded node
        
        Returns:
            Dictionary with iteration results
        """
        # Step 1: Select for human grading
        nodes_for_grading = self.step_human_grading(k)
        
        # Note: Steps 2-3 (human feedback) happen outside this method
        # The caller provides feedback via step_update_from_feedback()
        
        return {
            "nodes_for_grading": nodes_for_grading,
            "k": k,
            "m": m
        }
    
    def consume_feedback(
        self,
        human_feedback: List[tuple[TaxonomyNode, bool]]
    ):
        """
        Ingest human feedback into the project state (without triggering full iteration logic).
        Use this for incremental updates from contractor service.
        """
        # Step 2-3: Update tree from feedback
        self.step_update_from_feedback(human_feedback)
        
        # We don't persist here because ContractorService handles its own persistence for individual items.
        # But if we wanted to be safe, we could. For now, assuming ContractorService does the DB save.

    async def finalize_iteration(
        self,
        m: int = 5,
        num_children_per_node: int = 3
    ) -> dict:
        """
        Run the heavy lifting at the end of an iteration:
        - Derive rubrics
        - Rescore nodes
        - Expand tree
        """
        self._iteration_count += 1
        iteration = self._iteration_count
        
        # Step 4: Derive rubrics
        rubric, new_count, merged_count = await self.step_derive_rubrics()
        if rubric and self.storage:
            await self.persist_after_rubric_derivation(rubric, iteration)
        
        # Step 5: Re-score all nodes (only new principles)
        scoring_results = await self.step_rescore_all()
        
        # Persist score updates if storage is configured
        if self.storage:
            all_nodes = self.tree_manager.get_all_nodes()
            
            # Safety Sync: Ensure all nodes are definitely in DB before saving scores
            # This prevents ForeignKeyViolationError if a previous save failed or was raced
            if all_nodes:
                await self.storage.save_nodes_batch(all_nodes)
                await self.storage.flush()

            node_scores = [
                (node.id, node.rubric_score, node.ucb_score)
                for node in all_nodes
            ]
            await self.persist_after_scoring(node_scores)
            
            # Cache only NEW principle scores
            if scoring_results and rubric:
                principle_ids = [p.id for p in rubric.principles if p.id]
                node_principle_scores = {}
                # ... (same caching logic) ...
                if principle_ids:
                    node_ids = [node.id for node in all_nodes]
                    async with self.storage.database.get_session() as session:
                        from storage.repositories import PrincipleScoreRepository
                        repo = PrincipleScoreRepository(session)
                        cached_scores = await repo.get_cached_scores(node_ids, principle_ids)
                    
                    for node_id, (score, principle_scores) in scoring_results.items():
                        node_cached = cached_scores.get(node_id, {})
                        new_scores = [
                            ps for ps in principle_scores
                            if not ps.principle.id or ps.principle.id not in node_cached
                        ]
                        if new_scores:
                            node_principle_scores[node_id] = new_scores
                else:
                    for node_id, (score, principle_scores) in scoring_results.items():
                        node_principle_scores[node_id] = principle_scores
                
                if node_principle_scores:
                    await self.persist_principle_scores(node_principle_scores, iteration)
            
            # --- CALCULATE METRICS (Moved AFTER scoring) ---
            try:
                import statistics
                from data_models import RubricMetrics
                
                # 1. Variance & Average
                # Use ALL nodes for variance now? Or still just alive ones?
                # Usually variance of *active candidates* is what matters for selection.
                # But for rubric quality, maybe all nodes. Let's stick to alive for variance to measure "current population spread"
                # converting to all_nodes for alignment though is critical.
                
                alive_nodes = self.tree_manager.get_alive_nodes()
                scores = [n.rubric_score for n in alive_nodes]
                
                avg_score = statistics.median(scores) if scores else 0.0
                variance = statistics.variance(scores) if len(scores) > 1 else 0.0
                
                # 2. Alignment (Robust Median Gap)
                # Now that ALL nodes are rescored, these scores are fresh!
                relevant_scores = [s.node.rubric_score for s in self.project.human_labeled_samples if s.label]
                irrelevant_scores = [s.node.rubric_score for s in self.project.human_labeled_samples if not s.label]
                
                alignment = None
                if relevant_scores and irrelevant_scores:
                    median_rel = statistics.median(relevant_scores)
                    median_irrel = statistics.median(irrelevant_scores)
                    alignment = median_rel - median_irrel
                
                metrics = RubricMetrics(
                    num_principles=len(rubric.principles),
                    new_principles_count=new_count,
                    merged_principles_count=merged_count,
                    avg_score=avg_score,
                    score_variance=variance,
                    score_alignment=alignment
                )
                
                await self.storage.save_rubric_metrics(metrics, rubric.id)
                logger.info(f"Rubric Metrics Saved: Var={variance:.2f}, Alignment={alignment}, Merged={merged_count}")
            except Exception as e:
                logger.error(f"Failed to calculate metrics: {e}")
        
        # Step 6: Expand
        new_nodes = await self.step_expand(m, num_children_per_node)
        
        # Final state sync
        if self.storage:
            await self.save_state()
            await self.flush_storage()
        
        return {
            "rubric_updated": rubric is not None,
            "new_nodes": new_nodes,
            "total_labeled": len(self.project.human_labeled_samples),
            "total_rubrics": len(self.project.rubrics)
        }

    async def complete_iteration(
        self,
        human_feedback: List[tuple[TaxonomyNode, bool]],
        m: int = 5,
        num_children_per_node: int = 3
    ) -> dict:
        """
        Complete an iteration (Wrapper for backward compatibility).
        """
        # Ingest feedback
        self.consume_feedback(human_feedback)
        
        # Persist feedback explicitly (since complete_iteration implies full handling)
        if self.storage:
            # We don't have the iteration number fully resolved until finalize, 
            # but usually persistence happens before.
            # Let's match original behavior: persist feedback before finalize
            # The original saved samples with `iteration` but that variable was set from `self._iteration_count + 1`.
            # Let's do that.
            temp_iter = self._iteration_count + 1
            new_samples = [
                LabeledSample(node=node, label=label, iteration=temp_iter)
                for node, label in human_feedback
            ]
            await self.persist_after_feedback(new_samples)

        return await self.finalize_iteration(m, num_children_per_node)
    
    # ========================================================================
    # Async Persistence Methods (for storage integration)
    # ========================================================================
    
    async def save_state(self):
        """
        Save current state to storage (if storage is configured).
        
        This should be called after key operations to persist state.
        """
        if not self.storage:
            return
        
        # Save project (includes all nodes)
        await self.storage.save_project(self.project)
        
        # Save branch statistics
        branch_stats = list(self.tree_manager.get_branch_stats().values())
        if branch_stats:
            await self.storage.save_branch_stats_batch(branch_stats)
    
    async def persist_after_expansion(self, new_nodes: List[TaxonomyNode]):
        """
        Persist newly expanded nodes to storage.
        
        Args:
            new_nodes: List of newly created nodes
        """
        if not self.storage:
            return
        
        # Save new nodes in batch
        if new_nodes:
            await self.storage.save_nodes_batch(new_nodes)
    
    async def persist_after_scoring(self, node_scores: List[tuple]):
        """
        Persist score updates to storage.
        
        Args:
            node_scores: List of (node_id, rubric_score, ucb_score) tuples
        """
        if not self.storage:
            return
        
        # Update scores in batch
        if node_scores:
            await self.storage.update_scores_batch(node_scores)
    
    async def persist_after_feedback(self, labeled_samples: List[LabeledSample]):
        """
        Persist human feedback to storage.
        
        Args:
            labeled_samples: List of labeled samples
        """
        if not self.storage:
            return
        
        # Save each labeled sample
        for sample in labeled_samples:
            await self.storage.save_labeled_sample(sample)
        
        # Kill subtrees for irrelevant nodes (storage handles this efficiently with recursive CTE)
        for sample in labeled_samples:
            if not sample.label:  # Irrelevant - kill entire subtree
                await self.storage.kill_subtree(sample.node.id)
    
    async def persist_after_rubric_derivation(self, rubric: Rubric, iteration: int):
        """
        Persist newly derived rubric to storage.
        
        Args:
            rubric: Newly derived rubric
            iteration: Current iteration number
        """
        if not self.storage:
            return
        
        await self.storage.save_rubric(rubric, iteration)
    
    async def persist_principle_scores(
        self,
        node_principle_scores: dict[str, List['PrincipleScore']],
        rubric_version: int
    ):
        """
        Cache principle scores to storage.
        
        Args:
            node_principle_scores: Dict mapping node_id to list of PrincipleScore
            rubric_version: Rubric version number
        """
        if not self.storage:
            return
        
        await self.storage.save_principle_scores_batch(node_principle_scores, rubric_version)
    
    async def flush_storage(self):
        """Flush all pending storage operations."""
        if self.storage:
            await self.storage.flush()
    
    # ========================================================================
    # Async Wrappers with Persistence
    # ========================================================================
    
    async def load_from_storage(self, project_id: Optional[str] = None) -> bool:
        """
        Load project from storage and update all components.
        
        Args:
            project_id: Project ID to load (uses storage's default if None)
        
        Returns:
            True if project was loaded, False if not found
        """
        if not self.storage:
            return False
        
        loaded_project = await self.storage.load_project(project_id)
        if not loaded_project:
            return False
        
        # Update project
        self.project = loaded_project
        
        # Reinitialize tree manager with loaded root
        self.tree_manager = TreeManager(self.project.taxonomy_root)
        self.expansion_selector.tree_manager = self.tree_manager
        self.human_grading_sampler.tree_manager = self.tree_manager
        
        return True
    
