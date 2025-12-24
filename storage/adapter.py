"""
Async storage adapter with batching and caching.
"""
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager

from data_models import (
    Project, TaxonomyNode, LabeledSample, Rubric, PrincipleScore, BranchStats,
    RubricMetrics
)
from storage.database import Database
from storage.repositories import (
    NodeRepository, ProjectRepository, RubricRepository,
    LabeledSampleRepository, PrincipleScoreRepository, BranchStatsRepository,
    RubricMetricsRepository
)


class AsyncStorageAdapter:
    """
    Async storage adapter with efficient batching.
    """
    
    def __init__(self, database: Database, project_id: str = 'default', batch_size: int = 100):
        """
        Args:
            database: Database instance
            project_id: Project ID
            batch_size: Batch size for queued operations
        """
        self.database = database
        self.project_id = project_id
        self.batch_size = batch_size
        
        # Queues for batching
        self._pending_nodes: List[TaxonomyNode] = []
        self._pending_scores: List[Tuple[str, float, float]] = []
        self._pending_statuses: List[Tuple[str, str]] = []
    
    @asynccontextmanager
    async def session(self):
        """Get async session with transaction management."""
        async with self.database.get_session() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise
    
    async def save_project(self, project: Project):
        """Save project and flush all pending operations."""
        async with self.session() as s:
            repo = ProjectRepository(s)
            await repo.save(project)
        
        # Flush pending
        await self.flush()
    
    async def load_project(self, project_id: Optional[str] = None) -> Optional[Project]:
        """Load project."""
        pid = project_id or self.project_id
        async with self.session() as s:
            repo = ProjectRepository(s)
            return await repo.load(pid)
    
    async def clear_project(self, project_id: Optional[str] = None):
        """Clear all project data."""
        pid = project_id or self.project_id
        async with self.session() as s:
            repo = ProjectRepository(s)
            await repo.clear_project(pid)

    async def get_all_projects(self) -> List[dict]:
        """Get list of all projects."""
        async with self.session() as s:
            repo = ProjectRepository(s)
            return await repo.get_all_projects()
            
    async def delete_project(self, project_id: str):
        """Delete a project and all its data."""
        async with self.session() as s:
            repo = ProjectRepository(s)
            await repo.delete_project(project_id)
            await s.commit()

    async def save_nodes_batch(self, nodes: List[TaxonomyNode]):
        """Queue nodes for batch save."""
        self._pending_nodes.extend(nodes)
        if len(self._pending_nodes) >= self.batch_size:
            await self._flush_nodes()
    
    async def update_scores_batch(self, node_scores: List[Tuple[str, float, float]]):
        """Queue score updates."""
        self._pending_scores.extend(node_scores)
        if len(self._pending_scores) >= self.batch_size:
            await self._flush_scores()
    
    async def update_status_batch(self, node_statuses: List[Tuple[str, str]]):
        """Queue status updates."""
        self._pending_statuses.extend(node_statuses)
        if len(self._pending_statuses) >= self.batch_size:
            await self._flush_statuses()
    
    async def save_labeled_sample(self, sample: LabeledSample):
        """Save labeled sample immediately."""
        async with self.session() as s:
            repo = LabeledSampleRepository(s)
            await repo.save(self.project_id, sample)
            await s.commit()
    
    async def save_rubric(self, rubric: Rubric, iteration: int):
        """Save rubric immediately."""
        async with self.session() as s:
            repo = RubricRepository(s)
            await repo.save_rubric(self.project_id, rubric, iteration)
            await s.commit()
    
    async def save_rubric_metrics(self, metrics: RubricMetrics, rubric_id: str):
        """Save rubric metrics."""
        async with self.session() as s:
            repo = RubricMetricsRepository(s)
            await repo.save(metrics, rubric_id)
            await s.commit()

    async def get_rubric_metrics(self, project_id: Optional[str] = None) -> List[dict]:
        """Get all rubric metrics."""
        pid = project_id or self.project_id
        async with self.session() as s:
            repo = RubricMetricsRepository(s)
            return await repo.get_all_metrics(pid)

            
    async def save_principle_scores_batch(
        self,
        node_scores: dict[str, List[PrincipleScore]],
        rubric_version: int
    ):
        """
        Cache principle scores for multiple nodes.
        
        Args:
            node_scores: Dict mapping node_id to list of PrincipleScore
            rubric_version: Rubric version number
        """
        async with self.session() as s:
            repo = PrincipleScoreRepository(s)
            for node_id, scores in node_scores.items():
                await repo.save_batch(node_id, scores, rubric_version)
            await s.commit()
    
    async def save_branch_stats_batch(self, stats: List[BranchStats]):
        """Save branch statistics."""
        async with self.session() as s:
            repo = BranchStatsRepository(s)
            await repo.save_batch(self.project_id, stats)
            await s.commit()
    
    async def get_alive_nodes(self, subtree_root_id: Optional[str] = None) -> List[TaxonomyNode]:
        """Get alive nodes efficiently from database."""
        async with self.session() as s:
            repo = NodeRepository(s)
            return await repo.get_alive_nodes(self.project_id, subtree_root_id)
    
    async def kill_subtree(self, node_id: str):
        """Kill entire subtree."""
        async with self.session() as s:
            repo = NodeRepository(s)
            await repo.kill_subtree(node_id)
            await s.commit()
    
    async def _flush_nodes(self):
        """Flush pending nodes."""
        if not self._pending_nodes:
            return
        
        async with self.session() as s:
            repo = NodeRepository(s)
            await repo.save_batch(self.project_id, self._pending_nodes)
            await s.commit()
        self._pending_nodes = []
    
    async def _flush_scores(self):
        """Flush pending scores."""
        if not self._pending_scores:
            return
        
        async with self.session() as s:
            repo = NodeRepository(s)
            await repo.update_scores_batch(self._pending_scores)
            await s.commit()
        self._pending_scores = []
    
    async def _flush_statuses(self):
        """Flush pending statuses."""
        if not self._pending_statuses:
            return
        
        async with self.session() as s:
            repo = NodeRepository(s)
            await repo.update_status_batch(self._pending_statuses)
            await s.commit()
        self._pending_statuses = []
    
    async def flush(self):
        """Flush all pending operations."""
        await self._flush_nodes()
        await self._flush_scores()
        await self._flush_statuses()
