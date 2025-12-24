"""
Async repositories for converting between dataclasses and ORM models.
"""
import uuid
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, text, case, bindparam
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import insert

from data_models import (
    Project, TaxonomyNode, NodeStatus, Rubric, RubricPrinciple,
    LabeledSample, BranchStats, PrincipleScore
)
from storage.models import (
    ProjectModel, NodeModel, RubricModel, RubricPrincipleModel,
    LabeledSampleModel, PrincipleScoreModel, BranchStatsModel
)


class NodeRepository:
    """Repository for node operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_batch(self, project_id: str, nodes: List[TaxonomyNode]):
        """Save multiple nodes in batch with upsert."""
        if not nodes:
            return
            
        # Convert to dictionaries for insert
        # IMPORTANT: Deduplicate nodes by ID to prevent CardinalityViolationError
        # (Postgres ON CONFLICT fails if the batch itself has duplicates)
        seen_ids = set()
        node_dicts = []
        for node in nodes:
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            
            orm = self._to_orm(project_id, node)
            node_dicts.append({
                'id': orm.id,
                'project_id': orm.project_id,
                'parent_id': orm.parent_id,
                'content': orm.content,
                'depth': orm.depth,
                'created_iteration': orm.created_iteration,
                'status': orm.status,
                'human_label': orm.human_label,
                'rubric_score': orm.rubric_score,
                'visit_count': orm.visit_count,
                'ucb_score': orm.ucb_score
            })
            
        stmt = insert(NodeModel).values(node_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_=dict(
                parent_id=stmt.excluded.parent_id,
                content=stmt.excluded.content,
                depth=stmt.excluded.depth,
                created_iteration=stmt.excluded.created_iteration,
                status=stmt.excluded.status,
                human_label=stmt.excluded.human_label,
                rubric_score=stmt.excluded.rubric_score,
                visit_count=stmt.excluded.visit_count,
                ucb_score=stmt.excluded.ucb_score,
                updated_at=text('CURRENT_TIMESTAMP')
            )
        )
        await self.session.execute(stmt)
    
    async def update_scores_batch(self, node_scores: List[Tuple[str, float, float]]):
        """
        Batch update scores using bulk_update_mappings.
        
        Args:
            node_scores: List of (node_id, rubric_score, ucb_score) tuples
        """
        if not node_scores:
            return
        
        # ORM Bulk Update by Primary Key
        # We must NOT use a WHERE clause for this mode
        stmt = update(NodeModel)
        
        # Prepare params matching attribute names
        params = [
            {'id': nid, 'rubric_score': score, 'ucb_score': ucb}
            for nid, score, ucb in node_scores
        ]
        
        await self.session.execute(stmt, params, execution_options={"synchronize_session": None})
    
    async def update_status_batch(self, node_statuses: List[Tuple[str, str]]):
        """
        Batch update statuses using bulk_update_mappings.
        
        Args:
            node_statuses: List of (node_id, status) tuples
        """
        if not node_statuses:
            return
        
        # ORM Bulk Update by Primary Key
        stmt = update(NodeModel)
        
        # Prepare params matching attribute names
        params = [
            {'id': nid, 'status': status}
            for nid, status in node_statuses
        ]
        
        await self.session.execute(stmt, params, execution_options={"synchronize_session": None})
    
    async def kill_subtree(self, node_id: str):
        """Kill entire subtree using recursive CTE."""
        stmt = text("""
            WITH RECURSIVE descendants AS (
                SELECT id FROM nodes WHERE id = :root_id
                UNION ALL
                SELECT n.id FROM nodes n
                INNER JOIN descendants d ON n.parent_id = d.id
            )
            UPDATE nodes
            SET status = 'killed', updated_at = CURRENT_TIMESTAMP
            WHERE id IN (SELECT id FROM descendants)
        """)
        await self.session.execute(stmt, {'root_id': node_id})
    
    async def get_alive_nodes(
        self,
        project_id: str,
        subtree_root_id: Optional[str] = None
    ) -> List[TaxonomyNode]:
        """Get all alive nodes efficiently."""
        if subtree_root_id:
            # Recursive CTE for subtree - get node IDs first
            stmt = text("""
                WITH RECURSIVE descendants AS (
                    SELECT id FROM nodes WHERE id = :root_id
                    UNION ALL
                    SELECT n.id FROM nodes n
                    INNER JOIN descendants d ON n.parent_id = d.id
                )
                SELECT id FROM descendants
            """)
            result = await self.session.execute(stmt, {'root_id': subtree_root_id})
            descendant_ids = [row[0] for row in result.fetchall()]
            
            # Now load the actual nodes
            if descendant_ids:
                stmt = select(NodeModel).where(
                    NodeModel.id.in_(descendant_ids),
                    NodeModel.project_id == project_id,
                    NodeModel.status == 'alive'
                )
                result = await self.session.execute(stmt)
                orm_nodes = result.scalars().all()
            else:
                orm_nodes = []
        else:
            stmt = select(NodeModel).where(
                NodeModel.project_id == project_id,
                NodeModel.status == 'alive'
            )
            result = await self.session.execute(stmt)
            orm_nodes = result.scalars().all()
        
        return [self._to_dataclass(n) for n in orm_nodes]
    
    async def load_tree(self, project_id: str) -> Optional[TaxonomyNode]:
        """Load entire tree recursively."""
        # Load root
        stmt = select(NodeModel).where(
            NodeModel.project_id == project_id,
            NodeModel.parent_id == None
        ).options(selectinload(NodeModel.children))
        result = await self.session.execute(stmt)
        root_orm = result.scalar_one_or_none()
        
        if not root_orm:
            return None
        
        # Recursively load children
        await self._load_children_recursive(root_orm)
        return self._to_dataclass(root_orm)
    
    async def _load_children_recursive(self, parent_orm: NodeModel):
        """Recursively load all descendants."""
        stmt = select(NodeModel).where(
            NodeModel.parent_id == parent_orm.id
        ).options(selectinload(NodeModel.children))
        result = await self.session.execute(stmt)
        children = result.scalars().all()
        
        parent_orm.children = list(children)
        for child in children:
            await self._load_children_recursive(child)
    
    def _to_orm(self, project_id: str, node: TaxonomyNode) -> NodeModel:
        """Convert dataclass to ORM."""
        return NodeModel(
            id=node.id,
            project_id=project_id,
            parent_id=node.parent.id if node.parent else None,
            content=node.content,
            depth=node.depth,
            created_iteration=node.created_iteration,
            status=node.status.value,
            human_label=node.human_label,
            rubric_score=node.rubric_score,
            visit_count=node.visit_count,
            ucb_score=node.ucb_score
        )
    
    def _to_dataclass(self, orm: NodeModel) -> TaxonomyNode:
        """Convert ORM to dataclass."""
        node = TaxonomyNode(
            id=orm.id,
            content=orm.content,
            depth=orm.depth,
            created_iteration=getattr(orm, 'created_iteration', 0),
            status=NodeStatus(orm.status),
            human_label=orm.human_label,
            rubric_score=orm.rubric_score or 0.0,
            visit_count=orm.visit_count or 0,
            ucb_score=orm.ucb_score or 0.0,
            children=[]
        )
        
        # Convert children if loaded
        if 'children' in orm.__dict__ and orm.children:
            node.children = [self._to_dataclass(child) for child in orm.children]
            for child in node.children:
                child.parent = node
        
        return node


class ProjectRepository:
    """Repository for project operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.node_repo = NodeRepository(session)
    
    async def save(self, project: Project):
        """Save project."""
        project_id = getattr(project, 'id', 'default')
        
        # Upsert project
        stmt = insert(ProjectModel).values(
            id=project_id,
            product_description=project.product_description,
            new_branch_count=project.new_branch_count,
            total_expansions=project.total_expansions
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_=dict(
                product_description=stmt.excluded.product_description,
                new_branch_count=stmt.excluded.new_branch_count,
                total_expansions=stmt.excluded.total_expansions
            )
        )
        await self.session.execute(stmt)
        
        # Save nodes (batch)
        all_nodes = self._collect_all_nodes(project.taxonomy_root)
        if all_nodes:
            await self.node_repo.save_batch(project_id, all_nodes)
    
    async def clear_project(self, project_id: str):
        """Delete all data for a project."""
        # Delete in reverse order of dependencies
        
        # 1. Branch stats
        await self.session.execute(
            text("DELETE FROM branch_stats WHERE project_id = :pid"),
            {"pid": project_id}
        )
        
        # 2. Principle scores (via node_id in nodes, but better to use node_id match)
        # However, principle_scores doesn't have project_id directly, it links to nodes.
        # We can delete by join or just rely on cascade if configured. 
        # For safety/explicitness without relying on cascade:
        await self.session.execute(
            text("""
                DELETE FROM principle_scores 
                WHERE node_id IN (SELECT id FROM nodes WHERE project_id = :pid)
            """),
            {"pid": project_id}
        )
        
        # 3. Labeled samples
        await self.session.execute(
            text("DELETE FROM labeled_samples WHERE project_id = :pid"),
            {"pid": project_id}
        )
        
        # 4. Rubric values/principles
        # Principles
        await self.session.execute(
            text("""
                DELETE FROM rubric_principles 
                WHERE rubric_id IN (SELECT id FROM rubrics WHERE project_id = :pid)
            """),
            {"pid": project_id}
        )
        # Rubrics
        await self.session.execute(
            text("DELETE FROM rubrics WHERE project_id = :pid"),
            {"pid": project_id}
        )
        
        # 5. Nodes
        await self.session.execute(
            text("DELETE FROM nodes WHERE project_id = :pid"),
            {"pid": project_id}
        )
        
        # 6. Project itself (optional, depending on if we want to keep the entry)
        # Usually we keep the project record but clear its data, but instructions imply "reset".
        # Let's keep the project row but clear its stats.
        await self.session.execute(
            text("""
                UPDATE projects 
                SET total_expansions = 0, new_branch_count = 0 
                WHERE id = :pid
            """),
            {"pid": project_id}
        )

        
    async def delete_project(self, project_id: str):
        """Delete project and all associated data."""
        # Using cascade delete from database, so deleting project should fail everything
        # But for safety/explicitness we can rely on CASCADE foreign keys defined in models.py
        await self.session.execute(
            text("DELETE FROM projects WHERE id = :pid"),
            {"pid": project_id}
        )
    
    async def load(self, project_id: str) -> Optional[Project]:
        """Load project by ID."""
        stmt = select(ProjectModel).where(ProjectModel.id == project_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
            
        return await self._model_to_entity(model)

    async def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects with basic metadata."""
        stmt = select(ProjectModel.id, ProjectModel.product_description, ProjectModel.created_at).order_by(ProjectModel.created_at.desc())
        result = await self.session.execute(stmt)
        return [
            {
                "id": row.id, 
                "product_description": row.product_description,
                "created_at": row.created_at
            } 
            for row in result
        ]
    
    async def _model_to_entity(self, project_orm: ProjectModel) -> Project:
        """Convert ProjectModel to Project entity, loading related data."""
        project_id = project_orm.id
        
        # Load tree
        root = await self.node_repo.load_tree(project_id)
        if not root:
            root = TaxonomyNode(id="root", content="root")
        
        # Load rubrics
        rubrics = await self._load_rubrics(project_id)
        
        # Load labeled samples
        samples = await self._load_labeled_samples(project_id)
        
        project = Project(
            product_description=project_orm.product_description,
            taxonomy_root=root,
            rubrics=rubrics,
            human_labeled_samples=samples,
            new_branch_count=project_orm.new_branch_count,
            total_expansions=project_orm.total_expansions
        )
        project.id = project_id
        return project
    
    def _collect_all_nodes(self, root: TaxonomyNode) -> List[TaxonomyNode]:
        """Collect all nodes from tree."""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._collect_all_nodes(child))
        return nodes
    
    async def _load_rubrics(self, project_id: str) -> List[Rubric]:
        """Load rubrics."""
        stmt = select(RubricModel).where(
            RubricModel.project_id == project_id
        ).order_by(RubricModel.iteration)
        result = await self.session.execute(stmt)
        rubric_orms = result.scalars().all()
        
        rubrics = []
        for rubric_orm in rubric_orms:
            # Load principles
            stmt = select(RubricPrincipleModel).where(
                RubricPrincipleModel.rubric_id == rubric_orm.id
            )
            result = await self.session.execute(stmt)
            principle_orms = result.scalars().all()
            
            principles = [
                RubricPrinciple(
                    id=p.id,
                    description=p.description,
                    weight=p.weight
                )
                for p in principle_orms
            ]
            rubrics.append(Rubric(
                id=rubric_orm.id,
                principles=principles
            ))
        
        return rubrics
    
    async def _load_labeled_samples(self, project_id: str) -> List[LabeledSample]:
        """Load labeled samples."""
        stmt = select(LabeledSampleModel).where(LabeledSampleModel.project_id == project_id)
        result = await self.session.execute(stmt)
        sample_orms = result.scalars().all()
        
        if not sample_orms:
            return []
            
        # Get all node IDs
        node_ids = [s.node_id for s in sample_orms]
        
        # Load all nodes with their children using the NodeRepository method which handles this correctly
        # Note: We need to use recursion to load children properly
        nodes_map = {}
        if node_ids:
            # We can use the load_tree approach or just load specific nodes
            # For simplicity and correctness with async, let's load these specific nodes with eager loading of children
            # However, since children are recursive, a single selectinload(NodeModel.children) only goes one level deep
            # But the _to_dataclass method recursively converts children, so we need recursive loading
            
            # Use get_alive_nodes but allow it to find any node by ID (we'll modify it slightly or just use a new query)
            stmt = select(NodeModel).where(
                NodeModel.id.in_(node_ids),
                NodeModel.project_id == project_id
            ).options(selectinload(NodeModel.children))
            result = await self.session.execute(stmt)
            nodes = result.scalars().all()
            
            # For each node, we also need to recursively load its children to avoid lazy load errors
            for node in nodes:
                 await self.node_repo._load_children_recursive(node)
                 nodes_map[node.id] = self.node_repo._to_dataclass(node)
        
        samples = []
        for sample_orm in sample_orms:
            if sample_orm.node_id in nodes_map:
                samples.append(LabeledSample(
                    node=nodes_map[sample_orm.node_id],
                    label=sample_orm.label,
                    iteration=sample_orm.iteration
                ))
        
        return samples


class RubricRepository:
    """Repository for rubric operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_rubric(self, project_id: str, rubric: Rubric, iteration: int):
        """Save rubric with principles."""
        rubric_id = f"rubric_{project_id}_{iteration}_{uuid.uuid4().hex[:8]}"
        
        # Prevent duplicate rubrics for the same iteration
        # Delete any existing one first
        await self.session.execute(
            delete(RubricModel).where(
                RubricModel.project_id == project_id,
                RubricModel.iteration == iteration
            )
        )
        
        # Save rubric
        rubric_orm = RubricModel(
            id=rubric_id,
            project_id=project_id,
            iteration=iteration
        )
        self.session.add(rubric_orm)
        await self.session.flush()
        
        # Update dataclass with ID
        rubric.id = rubric_id
        
        # Save principles and update dataclass objects with IDs
        for i, principle in enumerate(rubric.principles):
            principle_id = f"{rubric_id}_principle_{i}"
            principle_orm = RubricPrincipleModel(
                id=principle_id,
                rubric_id=rubric_id,
                description=principle.description,
                weight=principle.weight
            )
            self.session.add(principle_orm)
            # Update the dataclass object with the ID so it can be used immediately
            principle.id = principle_id
        
        await self.session.flush()


class LabeledSampleRepository:
    """Repository for labeled sample operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, project_id: str, sample: LabeledSample):
        """Save labeled sample."""
        sample_orm = LabeledSampleModel(
            id=sample.node.id,
            project_id=project_id,
            node_id=sample.node.id,
            label=sample.label,
            iteration=sample.iteration
        )
        stmt = insert(LabeledSampleModel).values(
            id=sample_orm.id,
            project_id=sample_orm.project_id,
            node_id=sample_orm.node_id,
            label=sample_orm.label,
            iteration=sample_orm.iteration
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['node_id'],
            set_=dict(
                label=stmt.excluded.label,
                iteration=stmt.excluded.iteration
            )
        )
        await self.session.execute(stmt)


class PrincipleScoreRepository:
    """Repository for principle score cache operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_batch(self, node_id: str, scores: List[PrincipleScore], rubric_version: int):
        """
        Cache principle scores for a node.
        
        Args:
            node_id: ID of the node these scores are for
            scores: List of principle scores
            rubric_version: Rubric version number (iteration number)
        """
        score_dicts = []
        for ps in scores:
            # Use principle ID if available (from loaded rubrics), otherwise look up by description
            if ps.principle.id:
                principle_id = ps.principle.id
            else:
                # Fallback: look up by description (for backwards compatibility)
                principle_id = await self._get_principle_id(ps.principle, rubric_version)
                if not principle_id:
                    # Principle not found - skip this score
                    continue
            
            score_dicts.append({
                'id': f"{node_id}_{principle_id}_{rubric_version}_{uuid.uuid4().hex[:8]}",
                'node_id': node_id,
                'principle_id': principle_id,
                'score': ps.score,
                'reasoning': ps.reasoning,
                'rubric_version': rubric_version
            })
        
        if score_dicts:
            stmt = insert(PrincipleScoreModel).values(score_dicts)
            stmt = stmt.on_conflict_do_update(
                index_elements=['node_id', 'principle_id', 'rubric_version'],
                set_=dict(
                    score=stmt.excluded.score,
                    reasoning=stmt.excluded.reasoning
                )
            )
            await self.session.execute(stmt)
    
    async def get_cached_scores(
        self,
        node_ids: List[str],
        principle_ids: List[str]
    ) -> dict[str, dict[str, Tuple[float, Optional[str]]]]:
        """
        Get cached principle scores for nodes.
        
        Args:
            node_ids: List of node IDs to get scores for
            principle_ids: List of principle IDs to check
        
        Returns:
            Dictionary mapping node_id to dict of principle_id -> (score, reasoning)
        """
        if not node_ids or not principle_ids:
            return {}
        
        stmt = select(PrincipleScoreModel).where(
            PrincipleScoreModel.node_id.in_(node_ids),
            PrincipleScoreModel.principle_id.in_(principle_ids)
        )
        result = await self.session.execute(stmt)
        score_orms = result.scalars().all()
        
        # Group by node_id, then by principle_id
        cached = {}
        for score_orm in score_orms:
            if score_orm.node_id not in cached:
                cached[score_orm.node_id] = {}
            cached[score_orm.node_id][score_orm.principle_id] = (
                score_orm.score,
                score_orm.reasoning
            )
        
        return cached
    
    async def _get_principle_id(self, principle: RubricPrinciple, rubric_version: int) -> Optional[str]:
        """
        Get principle ID from database by matching description.
        
        Searches across all rubrics (since principles accumulate).
        """
        # Find principle with matching description across all rubrics
        # We search by description since principles accumulate across iterations
        stmt = select(RubricPrincipleModel).where(
            RubricPrincipleModel.description == principle.description
        ).order_by(RubricPrincipleModel.created_at.desc()).limit(1)
        result = await self.session.execute(stmt)
        principle_orm = result.scalar_one_or_none()
        
        return principle_orm.id if principle_orm else None


class BranchStatsRepository:
    """Repository for branch statistics operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_batch(self, project_id: str, stats: List[BranchStats]):
        """Save branch statistics."""
        stats_dicts = []
        for stat in stats:
            stats_dicts.append({
                'id': stat.root_child.id,
                'project_id': project_id,
                'branch_node_id': stat.root_child.id,
                'total_visits': stat.total_visits,
                'total_nodes': stat.total_nodes,
                'killed_count': stat.killed_count,
                'verified_relevant_count': stat.verified_relevant_count
            })
        
        if stats_dicts:
            stmt = insert(BranchStatsModel).values(stats_dicts)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_=dict(
                    total_visits=stmt.excluded.total_visits,
                    total_nodes=stmt.excluded.total_nodes,
                    killed_count=stmt.excluded.killed_count,
                    verified_relevant_count=stmt.excluded.verified_relevant_count
                )
            )
            await self.session.execute(stmt)


class RubricMetricsRepository:
    """Repository for rubric metrics operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, metrics: 'RubricMetrics', rubric_id: str):
        """Save rubric metrics."""
        from storage.models import RubricMetricsModel
        
        offset = uuid.uuid4().hex[:8]
        metric_id = f"rm_{rubric_id}_{offset}"

        orm = RubricMetricsModel(
            id=metric_id,
            rubric_id=rubric_id,
            num_principles=metrics.num_principles,
            new_principles_count=metrics.new_principles_count,
            merged_principles_count=metrics.merged_principles_count,
            avg_score=metrics.avg_score,
            score_variance=metrics.score_variance,
            score_alignment=metrics.score_alignment
        )
        self.session.add(orm)

    async def get_all_metrics(self, project_id: str) -> List[dict]:
        """
        Get all rubric metrics for a project, ordered by iteration.
        """
        from storage.models import RubricMetricsModel, RubricModel
        
        stmt = select(
            RubricModel.iteration,
            RubricMetricsModel.num_principles,
            RubricMetricsModel.new_principles_count,
            RubricMetricsModel.merged_principles_count,
            RubricMetricsModel.avg_score,
            RubricMetricsModel.score_variance,
            RubricMetricsModel.score_alignment
        ).join(
            RubricModel, RubricMetricsModel.rubric_id == RubricModel.id
        ).where(
            RubricModel.project_id == project_id
        ).order_by(
            RubricModel.iteration
        )
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        return [
            {
                "iteration": row.iteration,
                "num_principles": row.num_principles,
                "new_principles_count": row.new_principles_count,
                "merged_principles_count": row.merged_principles_count,
                "avg_score": row.avg_score,
                "score_variance": row.score_variance,
                "score_alignment": row.score_alignment
            }
            for row in rows
        ]
