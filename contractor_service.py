import logging
import asyncio
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import GradingQueueItem, NodeModel, ProjectModel, LabeledSampleModel
from storage.adapter import AsyncStorageAdapter
from human_grading_sampler import HumanGradingSampler
from orchestrator import TestSuiteOrchestrator

logger = logging.getLogger("contractor_service")

class ContractorService:
    def __init__(self, storage: AsyncStorageAdapter, sampler: HumanGradingSampler, orchestrator: TestSuiteOrchestrator):
        self.storage = storage
        self.sampler = sampler
        self.orchestrator = orchestrator
        self.assignment_timeout = timedelta(minutes=30)  # Configurable timeout
        self.refill_lock = asyncio.Lock()
        
    async def refill_queue(self, limit: int = 50):
        """
        Populate the grading queue with new samples.
        Only adds nodes that are not already in the queue or labeled.
        """
        async with self.refill_lock:
            async with self.storage.session() as session:
                # 1. Check current queue size
                stmt = select(func.count(GradingQueueItem.id)).where(
                    GradingQueueItem.project_id == self.storage.project_id,
                    GradingQueueItem.status == 'PENDING'
                )
                result = await session.execute(stmt)
                current_pending = result.scalar() or 0
                
                needed = limit - current_pending
                if needed <= 0:
                    return 0
                    
                # 1.5 Get existing queue node IDs and labeled node IDs first (for exclusion)
                q_stmt = select(GradingQueueItem.node_id).where(
                    GradingQueueItem.project_id == self.storage.project_id
                )
                q_res = await session.execute(q_stmt)
                queued_node_ids = set(q_res.scalars().all())
                
                # Also exclude already labeled nodes (DB Source of Truth)
                l_stmt = select(LabeledSampleModel.node_id).where(
                    LabeledSampleModel.project_id == self.storage.project_id
                )
                l_res = await session.execute(l_stmt)
                labeled_node_ids = set(l_res.scalars().all())
                
                # Combine exclusions
                all_excluded_ids = list(queued_node_ids.union(labeled_node_ids))

                # 2. Get samples from sampler
                candidates = self.sampler.select_for_human_grading(k=needed, excluded_ids=all_excluded_ids)
                
                if not candidates:
                    return 0

                # Ensure these nodes are persisted before we queue them.
                await self.storage.save_nodes_batch(candidates)
                await self.storage.flush()
                
                # Add to queue using PostgreSQL's ON CONFLICT DO NOTHING to be ultra safe
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                
                added_count = 0
                queue_dicts = []
                for node in candidates:
                    queue_dicts.append({
                        "id": f"queue_{node.id}",
                        "project_id": self.storage.project_id,
                        "node_id": node.id,
                        "status": 'PENDING'
                    })
                
                if queue_dicts:
                    stmt = pg_insert(GradingQueueItem).values(queue_dicts)
                    stmt = stmt.on_conflict_do_nothing(index_elements=['id'])
                    res = await session.execute(stmt)
                    await session.commit()
                    return len(queue_dicts) # Approximate, but since we filtered it's mostly correct
                
                return 0

    async def get_next_batch(self, contractor_id: str = "anon", batch_size: int = 5, background_tasks=None) -> Dict:
        """
        Get a batch of tasks for the contractor.
        Explicitly handles releasing stale tasks first.
        """
        async with self.storage.session() as session:
            # 1. Lazy reaping: Release stale tasks
            timeout_cutoff = datetime.utcnow() - self.assignment_timeout
            
            stale_stmt = update(GradingQueueItem).where(
                GradingQueueItem.status == 'ASSIGNED',
                GradingQueueItem.assigned_at < timeout_cutoff
            ).values(
                status='PENDING',
                assigned_to=None,
                assigned_at=None
            )
            await session.execute(stale_stmt)

            # 1.5 Check for specific assignments for this contractor
            # If they already have tasks, return them (sticky session)
            existing_stmt = select(GradingQueueItem).where(
                GradingQueueItem.project_id == self.storage.project_id,
                GradingQueueItem.status == 'ASSIGNED',
                GradingQueueItem.assigned_to == contractor_id
            ).order_by(GradingQueueItem.assigned_at.asc())
            
            existing_result = await session.execute(existing_stmt)
            existing_items = existing_result.scalars().all()
            
            items = []
            if existing_items:
                items = list(existing_items)
                # If we have less than batch size, maybe we fill up? 
                # User requirement: "dont reassign the new tasks but give them their previous ones"
                # This suggests simpler logic: just give what they have.
                # But if they finished some, we might want to top up?
                # Let's stick to "give them their previous ones" strictly for now. 
                # If they want more, they finish these and ask again.
            else:
                # 2. Find next pending tasks (up to batch_size)
                stmt = select(GradingQueueItem).where(
                    GradingQueueItem.project_id == self.storage.project_id,
                    GradingQueueItem.status == 'PENDING'
                ).order_by(GradingQueueItem.created_at.asc()).limit(batch_size).with_for_update()
                
                result = await session.execute(stmt)
                items = result.scalars().all()
            
            if not items and not existing_items: # Trigger refill only if completely empty logic
                # Trigger refill if empty and we tried to fetch new ones
                # Trigger refill if empty
                if background_tasks:
                    background_tasks.add_task(self.process_auto_refill_if_needed)
                
                return {
                    "tasks": [],
                    "project_id": self.storage.project_id,
                    "product_description": self.orchestrator.project.product_description
                }
                
            # 3. Assign to contractor (if new)
            task_data = []
            now = datetime.utcnow()
            
            for item in items:
                # Only update status if it's new (PENDING)
                if item.status == 'PENDING':
                    item.status = 'ASSIGNED'
                    item.assigned_to = contractor_id
                    item.assigned_at = now
                
                # Fetch node content (could optimize with join but loop is small)
                # Use in-memory tree to get full path efficiently
                in_memory_node = self.orchestrator.tree_manager.get_node(item.node_id)
                full_path = ""
                content = "Unknown Node"
                
                if in_memory_node:
                    content = in_memory_node.content
                    # Build path: Root > ... > Parent > Node
                    chain = []
                    curr = in_memory_node
                    while curr and curr.depth > 0: # Skip root for display cleanliness
                        chain.append(curr.content)
                        curr = curr.parent
                    full_path = " > ".join(reversed(chain))
                else:
                    # Fallback to DB query if not in memory (shouldn't happen often, e.g. after restart)
                    # Use CTE to reconstruct path up to root
                    from sqlalchemy.orm import aliased
                    
                    recursive_cte = select(NodeModel).where(NodeModel.id == item.node_id).cte(name="node_ancestry", recursive=True)
                    parent_alias = aliased(NodeModel)
                    recursive_cte = recursive_cte.union_all(
                        select(parent_alias).join(recursive_cte, parent_alias.id == recursive_cte.c.parent_id)
                    )
                    
                    # Order by depth ASC (Root -> ... -> Node) excluding root for clean display
                    path_stmt = select(recursive_cte.c.content).where(recursive_cte.c.id != 'root').order_by(recursive_cte.c.depth.asc())
                    path_res = await session.execute(path_stmt)
                    path_parts = list(path_res.scalars().all())
                    
                    if path_parts:
                         full_path = " > ".join(path_parts)
                         content = path_parts[-1]
                    else:
                         # Use simple query if CTE fails or returns empty (e.g. only root?)
                         node_stmt = select(NodeModel).where(NodeModel.id == item.node_id)
                         node_res = await session.execute(node_stmt)
                         node = node_res.scalar_one()
                         content = node.content
                         full_path = node.content
                
                task_data.append({
                    "task_id": item.id,
                    "node_id": item.node_id,
                    "content": content,
                    "full_path_content": full_path
                })
            
            await session.commit()
            
            # Check if we need to refill after taking items
            if len(items) < batch_size or (len(items) == batch_size): 
                # Be proactive: check if we drained the queue
                if background_tasks:
                     background_tasks.add_task(self.process_auto_refill_if_needed)
            
            return {
                "tasks": task_data,
                "project_id": self.storage.project_id,
                "product_description": self.orchestrator.project.product_description
            }

    async def submit_grade(self, task_id: str, is_relevant: bool, contractor_id: str = "anon", background_tasks=None):
        """
        Submit a grade for a task.
        Returns: True if feedback loop triggered.
        """
        loop_triggered = False
        async with self.storage.session() as session:
            # 1. Verify task assignment
            stmt = select(GradingQueueItem).where(GradingQueueItem.id == task_id)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()
            
            if not item:
                raise ValueError("Task not found")
                
            if item.status != 'ASSIGNED':
                if item.status == 'PENDING':
                    # Accept it anyway
                    pass
                elif item.assigned_to != contractor_id and item.assigned_to is not None:
                     # Assigned to someone else
                   raise ValueError("Task assigned to another user")

            # 2. Save grading result
            # Check if already labeled
            existing_stmt = select(LabeledSampleModel).where(
                LabeledSampleModel.node_id == item.node_id
            )
            existing = (await session.execute(existing_stmt)).scalar_one_or_none()
            
            if not existing:
                # Add label
                new_label = LabeledSampleModel(
                    id=f"label_{item.node_id}",
                    project_id=item.project_id,
                    node_id=item.node_id,
                    label=is_relevant,
                    iteration=len(self.orchestrator.project.rubrics) # Approximate iteration
                )
                session.add(new_label)
                
                # Update Node status
                update_node = update(NodeModel).where(
                     NodeModel.id == item.node_id
                ).values(human_label=is_relevant)
                await session.execute(update_node)
            
            # 3. Remove from queue
            await session.delete(item)
            
            await session.commit()
            
            # 4. Update in-memory orchestrator (Incremental)
            # Find the node object in the tree manager or orchestrator
            # We need the actual node object, not just ID, because consume_feedback expects (node, label)
            # We can find it via tree manager
            in_memory_node = self.orchestrator.tree_manager.get_node(item.node_id)
            if in_memory_node:
                self.orchestrator.consume_feedback([(in_memory_node, is_relevant)])
            else:
                logger.warning(f"Node {item.node_id} not found in memory, skipping in-memory update")

            # 5. Check trigger condition
            loop_triggered = await self.check_and_trigger_loop()
        
        # 6. Auto-refill if needed (Background)
        # Threshold: if queue < 5, refill to ensure we have ~10
        if background_tasks:
            background_tasks.add_task(self.process_auto_refill_if_needed)
        else:
            # Fallback for sync contexts or tests
            await self.process_auto_refill_if_needed()
            
        return loop_triggered

    async def submit_batch_grades(self, submissions: List[dict], contractor_id: str = "anon", background_tasks=None) -> bool:
        """
        Submit a batch of grades.
        submissions: List of dicts with {'task_id': str, 'is_relevant': bool}
        Returns: True if feedback loop triggered.
        """
        loop_triggered = False
        feedback_buffer = []
        
        logger.info(f"Received batch submission with {len(submissions)} items")
        logger.info(f"Submissions content: {submissions}")

        async with self.storage.session() as session:
            for sub in submissions:
                task_id = sub['task_id']
                is_relevant = sub['is_relevant']
                
                # 1. Verify task assignment
                stmt = select(GradingQueueItem).where(GradingQueueItem.id == task_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    # Skip invalid tasks but log
                    logger.warning(f"Task {task_id} not found during batch submit")
                    continue
                    
                if item.status != 'ASSIGNED':
                    if item.status == 'PENDING':
                        pass
                    elif item.assigned_to != contractor_id and item.assigned_to is not None:
                         logger.warning(f"Task {task_id} assigned to another user")
                         continue

                # 2. Save grading result
                existing_stmt = select(LabeledSampleModel).where(
                    LabeledSampleModel.node_id == item.node_id
                )
                existing = (await session.execute(existing_stmt)).scalar_one_or_none()
                
                if not existing:
                    # Add label
                    new_label = LabeledSampleModel(
                        id=f"label_{item.node_id}",
                        project_id=item.project_id,
                        node_id=item.node_id,
                        label=is_relevant,
                        iteration=len(self.orchestrator.project.rubrics)
                    )
                    session.add(new_label)
                    
                    # Update Node status
                    update_node = update(NodeModel).where(
                         NodeModel.id == item.node_id
                    ).values(human_label=is_relevant)
                    await session.execute(update_node)
                    
                # Collect for in-memory update (Check if memory needs update regardless of DB state)
                in_memory_node = self.orchestrator.tree_manager.get_node(item.node_id)
                if in_memory_node:
                    # If memory is stale (no label) OR we just added it, update it.
                    # Note: consume_feedback is idempotent-ish (it appends to list), 
                    # so we should ideally check if it's already there to avoid duplicates in projection list.
                    # But for now, ensuring the Node object has the label is key.
                    if in_memory_node.human_label is None:
                        feedback_buffer.append((in_memory_node, is_relevant))
                    elif in_memory_node.human_label != is_relevant:
                        # Label changed? Update it.
                        feedback_buffer.append((in_memory_node, is_relevant))
                
                # 3. Remove from queue
                await session.delete(item)
            
            await session.commit()
            
            # 4. Update in-memory orchestrator (Batch)
            if feedback_buffer:
                logger.info(f"Sending {len(feedback_buffer)} items to consume_feedback")
                self.orchestrator.consume_feedback(feedback_buffer)
            else:
                logger.warning("Feedback buffer is empty! No in-memory updates.")
            
            # 5. Check trigger condition
            loop_triggered = await self.check_and_trigger_loop()
        
        # 6. Auto-refill if needed (Background)
        if background_tasks:
             background_tasks.add_task(self.process_auto_refill_if_needed)
        else:
             await self.process_auto_refill_if_needed()
             
        return loop_triggered
            
    async def process_auto_refill_if_needed(self):
        """Helper to check queue size and refill if needed."""
        try:
            async with self.storage.session() as session:
                 pending_stmt = select(func.count(GradingQueueItem.id)).where(
                    GradingQueueItem.project_id == self.storage.project_id,
                    GradingQueueItem.status == 'PENDING'
                )
                 pending_count = (await session.execute(pending_stmt)).scalar() or 0
            
            # Threshold: Keep at least 20 items buffer (or if queue is empty)
            if pending_count < 20:
                refill_target = 50 
                limit = refill_target - pending_count
                if limit > 0:
                    logger.info(f"Queue low ({pending_count}), triggering background refill of {limit} items...")
                    added = await self.refill_queue(limit=limit)
                    if added < limit:
                        logger.warning(f"Could only add {added} items. Queue size: {pending_count+added}. Triggering emergency expansion.")
                        
                        # Exhaustion detected! Force an expansion cycle.
                        try:
                            result = await self.orchestrator.finalize_iteration()
                            logger.info(f"Emergency expansion complete. New nodes: {len(result['new_nodes'])}")
                            
                            # Try refilling again with the new nodes
                            retry_added = await self.refill_queue(limit=limit - added)
                            logger.info(f"Added {retry_added} additional items after expansion.")
                        except Exception as e:
                            logger.error(f"Failed to recover from exhaustion: {e}")
        except Exception as e:
            logger.error(f"Error in background auto-refill: {e}")

    async def check_and_trigger_loop(self, threshold: int = 20):
        """
        Check if queue is running low (<= threshold).
        If yes, trigger orchestrator.finalize_iteration() to expand tree and generate new tasks.
        """
        async with self.storage.session() as session:
             # Check pending queue size
             stmt = select(func.count(GradingQueueItem.id)).where(
                GradingQueueItem.project_id == self.storage.project_id,
                GradingQueueItem.status == 'PENDING'
             )
             pending_count = (await session.execute(stmt)).scalar() or 0
             
             logger.info(f"Checking trigger: pending_queue={pending_count}, threshold={threshold}")

             if pending_count <= threshold:
                 logger.info(f"Queue low ({pending_count} <= {threshold}). Triggering feedback loop to expand tree!")
                 
                 # RUN ORCHESTRATOR LOGIC
                 try:
                     # Request 10 nodes for UCB selection to broaden the funnel
                     result = await self.orchestrator.finalize_iteration(m=10)
                     logger.info(f"Iteration finalized. New nodes: {len(result['new_nodes'])}")
                     
                     # Force refill to pull in the newly created nodes
                     await self.refill_queue(limit=10)
                 except Exception as e:
                     logger.error(f"Failed to finalize iteration: {e}")
                     return False
                     
                 return True
                 
        return False
