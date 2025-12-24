import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from contextlib import asynccontextmanager
from storage.adapter import AsyncStorageAdapter
from storage.database import Database
from data_models import Project, TaxonomyNode
from tree_manager import TreeManager
from expansion_selector import ExpansionSelector
from human_grading_sampler import HumanGradingSampler
from orchestrator import TestSuiteOrchestrator

from taxonomy_generator import TaxonomyGenerator, RealLLMTaxonomyGenerator
from rubric_scorer import RubricScorer, RealLLMScorer
from rubric_deriver import RubricDeriver, RealLLMRubricDeriver
from feedback_provider import FeedbackProvider, RandomFeedbackProvider, LLMFeedbackProvider
from feedback_provider import FeedbackProvider, RandomFeedbackProvider, LLMFeedbackProvider
from llm_client import LLMClient
from contractor_service import ContractorService
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("server")

# Removed in favor of lifespan
# app = FastAPI(title="Edge Case Sampler API", version="0.1.0")

# Middleware added in lifespan block
# app.add_middleware( ... )

# Global state
orchestrator: Optional[TestSuiteOrchestrator] = None
contractor_service: Optional[ContractorService] = None
db: Optional[Database] = None

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

async def setup_orchestrator(project_id: str, product_description: Optional[str] = None, reinitialize: bool = False, num_initial_topics: int = 3):
    global orchestrator, contractor_service, db
    
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
        
    storage = AsyncStorageAdapter(db, project_id=project_id)
    
    # Load or create project
    project = await storage.load_project()
    
    if reinitialize:
        logger.info(f"Reinitializing project {project_id} - clearing existing data")
        await storage.clear_project()
        project = None

    if not project and product_description:
        logger.info(f"Creating project: {project_id}")
        root = TaxonomyNode(id="root", content="root")
        project = Project(
            product_description=product_description or "Default description",
            taxonomy_root=root
        )
        project.id = project_id
    elif project:
        logger.info(f"Loaded existing project: {project_id}")
    else:
        logger.warning(f"Project {project_id} not found and no description provided.")
        return None

    # Initialize components with Real LLM
    logger.info("Initializing components with Real LLM implementations")
    llm_client = LLMClient()
    llm_taxonomy = RealLLMTaxonomyGenerator(llm_client)
    llm_scorer = RealLLMScorer(llm_client)
    llm_deriver = RealLLMRubricDeriver(llm_client)
    
    taxonomy_generator = TaxonomyGenerator(llm_taxonomy)
    rubric_scorer = RubricScorer(llm_scorer, top_k=2)
    rubric_deriver = RubricDeriver(llm_deriver)
    
    tree_manager = TreeManager(project.taxonomy_root)
    expansion_selector = ExpansionSelector(tree_manager, project)
    human_grading_sampler = HumanGradingSampler(tree_manager, expansion_selector)
    
    orchestrator = TestSuiteOrchestrator(
        project=project,
        taxonomy_generator=taxonomy_generator,
        rubric_scorer=rubric_scorer,
        rubric_deriver=rubric_deriver,
        expansion_selector=expansion_selector,
        human_grading_sampler=human_grading_sampler,
        tree_manager=tree_manager,
        storage=storage
    )
    
    contractor_service = ContractorService(storage, human_grading_sampler, orchestrator)
    
    # Run initialization logic if empty and just created
    # Check if we need to run initial generation: either it's a fresh re-init OR it's an empty existing project
    should_run_init = reinitialize or (len(project.taxonomy_root.children) == 0)
    
    if should_run_init:
        try:
            logger.info("Project empty. Running initial taxonomy generation...")
            await orchestrator.initialize(num_initial_topics=num_initial_topics)
        except Exception as e:
            logger.error(f"Failed to auto-generate initial topics: {e}")

    return orchestrator


# ----------------------------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------------------------

class InitRequest(BaseModel):
    project_id: str = "default_project"
    product_description: str
    num_initial_topics: int = 3
    reinitialize: bool = False

class ProjectListItem(BaseModel):
    id: str
    product_description: str
    created_at: Any

class ProjectListResponse(BaseModel):
    projects: List[ProjectListItem]
    current_project_id: str

class SwitchProjectRequest(BaseModel):
    project_id: str


class NodeResponse(BaseModel):
    id: str
    content: str
    full_path_content: str = ""
    rubric_score: Optional[float]
    ucb_score: Optional[float]
    depth: int

class IterationResponse(BaseModel):
    nodes_for_grading: List[NodeResponse]
    iteration_id: int

class FeedbackItem(BaseModel):
    node_id: str
    is_relevant: bool

class FeedbackRequest(BaseModel):
    items: List[FeedbackItem]

class StatsResponse(BaseModel):
    total_nodes: int
    total_rubrics: int
    total_labeled: int
    current_rubric_principles: List[str]

class TreeNodeResponse(BaseModel):
    id: str
    content: str
    depth: int
    status: str
    rubric_score: Optional[float]
    ucb_score: Optional[float]
    human_label: Optional[bool]  # True=relevant, False=irrelevant, None=unlabeled
    created_iteration: int = 0
    children: List['TreeNodeResponse'] = []

# Enable forward reference for recursive model
TreeNodeResponse.model_rebuild()

class RubricPrincipleResponse(BaseModel):
    id: str
    description: str
    weight: float

class RubricResponse(BaseModel):
    id: str
    iteration: int
    principles: List[RubricPrincipleResponse]

class NodePrincipleScoreResponse(BaseModel):
    principle_id: str
    score: float
    reasoning: Optional[str]

class NodeScoreResponse(BaseModel):
    node_id: str
    content: str
    full_path_content: str = ""
    aggregate_score: float
    principle_scores: List[NodePrincipleScoreResponse]

# ----------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://edgecase:dev_password@db:5432/edgecase_testing"
    )
    print(f"Connecting to database: {db_url}")
    db = Database(db_url)
    await db.create_tables()

    # Try auto-load most recent project
    try:
        storage = AsyncStorageAdapter(db)
        projects = await storage.get_all_projects()
        
        target_project_id = "default_project"
        description = "A smart home security camera with AI features."
        
        if projects:
            # Load the most recently created/updated project
            # (projects list is already ordered by created_at desc in repo)
            latest = projects[0]
            target_project_id = latest['id']
            description = latest['product_description']
            logger.info(f"Auto-loading most recent project: {target_project_id}")
        else:
            logger.info(f"No existing projects found. Creating default: {target_project_id}")

        await setup_orchestrator(
            target_project_id,
            product_description=description,
            reinitialize=False # NEVER reinitialize on auto-load
        )
    except Exception as e:
        logger.warning(f"Could not auto-load project: {e}")

    yield
    
    # Shutdown
    if db:
        await db.close()

app = FastAPI(title="Edge Case Sampler API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

@app.post("/init")
async def initialize_project(request: InitRequest):
    global orchestrator, db
    
    # Check if project exists to prevent accidental overwrite
    if not request.reinitialize and db:
        storage = AsyncStorageAdapter(db)
        projects = await storage.get_all_projects()
        if any(p['id'] == request.project_id for p in projects):
             raise HTTPException(status_code=400, detail=f"Project '{request.project_id}' already exists. Please use a different name or enable reinitialization explicitly.")
    
    res = await setup_orchestrator(
        project_id=request.project_id,
        product_description=request.product_description,
        reinitialize=request.reinitialize,
        num_initial_topics=request.num_initial_topics
    )
    
    if not res:
        raise HTTPException(status_code=500, detail="Failed to initialize project")
    
    project = orchestrator.project
    logger.info(f"Project initialized. Total nodes: {len(project.taxonomy_root.children)}")
    return {"status": "initialized", "project_id": request.project_id, "nodes": len(project.taxonomy_root.children)}

@app.get("/projects", response_model=ProjectListResponse)
async def list_projects():
    global orchestrator, db
    
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
        
    # We can use a temporary adapter just to list projects if orchestrator isn't set, 
    # but orchestrator usually is set. 
    # Better to use a fresh adapter to be safe or reuse current storage.
    
    storage = AsyncStorageAdapter(db)
    projects = await storage.get_all_projects()
    
    current_id = orchestrator.project.id if orchestrator else "none"
    
    return ProjectListResponse(
        projects=[
            ProjectListItem(
                id=p["id"],
                product_description=p["product_description"],
                created_at=p["created_at"]
            ) for p in projects
        ],
        current_project_id=current_id
    )

@app.post("/project/switch")
async def switch_project(request: SwitchProjectRequest):
    global orchestrator
    
    # Check if project exists first? setup_orchestrator handles it.
    # If project doesn't exist, setup_orchestrator will fail unless description is provided.
    # Here we assume we are switching to an EXISTING project.
    
    res = await setup_orchestrator(
        project_id=request.project_id,
        reinitialize=False
    )
    
    if not res:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")
        
    return {"status": "switched", "project_id": request.project_id}

@app.delete("/project/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    global db, orchestrator
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
        
    try:
        storage = AsyncStorageAdapter(db)
        await storage.delete_project(project_id)
        
        if orchestrator and orchestrator.project.id == project_id:
             logger.warning(f"Active project {project_id} was deleted.")
        
        return {"status": "ok", "message": f"Project {project_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/iteration/next", response_model=IterationResponse)
async def next_iteration(k: int = 5, m: int = 3):
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized. Call /init first.")
    
    result = orchestrator.run_iteration(k=k, m=m)
    nodes = result["nodes_for_grading"]
    
    node_responses = [
        NodeResponse(
            id=n.id,
            content=n.content,
            full_path_content=n.get_full_path_content(),
            rubric_score=n.rubric_score,
            ucb_score=n.ucb_score,
            depth=n.depth
        ) for n in nodes
    ]
    
    # In a real app we'd track iteration ID properly
    return IterationResponse(
        nodes_for_grading=node_responses,
        iteration_id=len(orchestrator.project.rubrics) + 1
    )

@app.post("/iteration/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized. Call /init first.")
    
    # Map feedback to (node, bool) tuples
    # We need to find the node object in the tree
    processed_feedback = []
    
    # Helper to find node by ID (BFS)
    def find_node(node_id: str) -> Optional[TaxonomyNode]:
        queue = [orchestrator.project.taxonomy_root]
        while queue:
            curr = queue.pop(0)
            if curr.id == node_id:
                return curr
            queue.extend(curr.children)
        return None

    for item in feedback.items:
        node = find_node(item.node_id)
        if node:
            processed_feedback.append((node, item.is_relevant))
        else:
            print(f"Warning: Node {item.node_id} not found for feedback")
            
    await orchestrator.complete_iteration(processed_feedback)
    
    return {"status": "iteration_complete"}

@app.get("/state", response_model=StatsResponse)
async def get_state():
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized. Call /init first.")
    
    rubric = orchestrator.project.get_current_rubric()
    principles = [p.description for p in rubric.principles] if rubric else []
    
    # Count total nodes
    count = 0
    queue = [orchestrator.project.taxonomy_root]
    while queue:
        curr = queue.pop(0)
        count += 1
        queue.extend(curr.children)

    return StatsResponse(
        total_nodes=count,
        total_rubrics=len(orchestrator.project.rubrics),
        total_labeled=len(orchestrator.project.human_labeled_samples),
        current_rubric_principles=principles
    )

@app.get("/tree", response_model=TreeNodeResponse)
async def get_tree():
    """Get the full taxonomy tree with scores and labels."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized. Call /init first.")
    
    # Build a set of labeled node IDs for quick lookup
    labeled_nodes = {sample.node.id: sample.label for sample in orchestrator.project.human_labeled_samples}
    
    def convert_node(node: TaxonomyNode) -> TreeNodeResponse:
        return TreeNodeResponse(
            id=node.id,
            content=node.content,
            depth=node.depth,
            status=node.status.value,
            rubric_score=node.rubric_score,
            ucb_score=node.ucb_score,
            # Prefer dictionary lookup (Project source of truth), but fallback to Node property (DB source of truth)
            human_label=labeled_nodes.get(node.id, node.human_label),
            created_iteration=node.created_iteration,
            children=[convert_node(child) for child in node.children]
        )
    
    return convert_node(orchestrator.project.taxonomy_root)

class AutoIterationRequest(BaseModel):
    k: int = 5
    m: int = 3
    mode: str = "random"  # or "llm"
    iterations: int = 1

# Simple in-memory job store and lock
background_jobs = {}
import asyncio
import uuid
iteration_lock = asyncio.Lock()

async def run_iterations_background(task_id: str, request: AutoIterationRequest):
    """Background task wrapper for running iterations."""
    # Acquire lock to prevent concurrent runs
    if iteration_lock.locked():
        background_jobs[task_id] = {"status": "failed", "error": "Another iteration is already running"}
        return

    # Double check locking with context manager
    async with iteration_lock:
        background_jobs[task_id] = {"status": "running", "results": []}
        try:
            # Setup feedback provider
            if request.mode == "llm":
                logger.info("Using LLM Feedback Provider")
                llm_client = LLMClient()
                feedback_provider = LLMFeedbackProvider(llm_client)
            else:
                logger.info("Using Random Feedback Provider")
                feedback_provider = RandomFeedbackProvider(acceptance_rate=0.5)
                
            results = []
            
            for i in range(request.iterations):
                logger.info(f"--- Auto Iteration {i+1}/{request.iterations} ---")
                
                # 1. Get nodes (Sync/CPU bound - might block briefly)
                iter_result = orchestrator.run_iteration(k=request.k, m=request.m)
                nodes = iter_result["nodes_for_grading"]
                
                if not nodes:
                     logger.warning("No nodes for grading, stopping.")
                     break
                     
                # 2. Get feedback for ALL nodes in one batch (Async - yields well)
                logger.info(f"Getting feedback for {len(nodes)} nodes (Provider: {type(feedback_provider).__name__})")
                
                batch_results = await feedback_provider.get_feedback_batch(
                    nodes, 
                    orchestrator.project.product_description
                )
                
                # Zip results back to nodes
                feedback_data = []
                for node, (is_relevant, _) in zip(nodes, batch_results):
                     feedback_data.append((node, is_relevant))
                    
                # 3. Complete iteration (Mixed Async/Sync)
                completion_result = await orchestrator.complete_iteration(feedback_data, m=request.m)
                results.append({
                    "iteration": i + 1,
                    "new_nodes": len(completion_result["new_nodes"]),
                    "rubric_updated": completion_result["rubric_updated"]
                })
                
                # Update progress
                background_jobs[task_id]["results"] = results
                
            background_jobs[task_id]["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Background iteration failed: {e}", exc_info=True)
            background_jobs[task_id]["status"] = "failed"
            background_jobs[task_id]["error"] = str(e)

@app.post("/iteration/auto")
async def auto_run_iteration(request: AutoIterationRequest, background_tasks: BackgroundTasks):
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized. Call /init first.")
    
    # Fast check before even accepting
    if iteration_lock.locked():
        raise HTTPException(status_code=409, detail="An iteration job is already running")
        
    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_iterations_background, task_id, request)
    return {"status": "started", "task_id": task_id}

@app.get("/iteration/auto/status/{task_id}")
async def get_auto_iteration_status(task_id: str):
    job = background_jobs.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    return job

# ----------------------------------------------------------------------------
# Contractor Endpoints
# ----------------------------------------------------------------------------

class ContractorTask(BaseModel):
    task_id: str
    node_id: str
    content: str
    full_path_content: str
    
class ContractorBatchResponse(BaseModel):
    tasks: List[ContractorTask]
# ... (rest of the file)
    product_description: str | None
    project_id: str | None
    message: str | None = None

class ContractorSubmitRequest(BaseModel):
    is_relevant: bool
    contractor_id: str = "anon"

class ContractorBatchSubmitRequest(BaseModel):
    submissions: List[dict] # List of {"task_id": "...", "is_relevant": bool}
    contractor_id: str = "anon"

class QueueRefillRequest(BaseModel):
    limit: int = 50

@app.post("/contractor/queue/fill")
async def fill_queue(request: QueueRefillRequest):
    global contractor_service
    if not contractor_service:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    
    added = await contractor_service.refill_queue(limit=request.limit)
    return {"status": "ok", "added": added}

@app.get("/contractor/next", response_model=ContractorBatchResponse)
async def get_next_batch(background_tasks: BackgroundTasks, contractor_id: str = "anon"):
    global contractor_service
    if not contractor_service:
        raise HTTPException(status_code=400, detail="Project not initialized.")
        
    data = await contractor_service.get_next_batch(contractor_id, batch_size=5, background_tasks=background_tasks)
    
    tasks = [
        ContractorTask(
            task_id=t["task_id"],
            node_id=t["node_id"],
            content=t["content"],
            full_path_content=t.get("full_path_content", t["content"])
        ) for t in data["tasks"]
    ]
    
    if not tasks:
        return ContractorBatchResponse(
            tasks=[],
            product_description=data["product_description"],
            project_id=data["project_id"],
            message="No tasks available"
        )
        
    return ContractorBatchResponse(
        tasks=tasks,
        product_description=data["product_description"],
        project_id=data["project_id"]
    )

@app.post("/contractor/task/{task_id}/submit")
async def submit_contractor_grade(task_id: str, request: ContractorSubmitRequest, background_tasks: BackgroundTasks):
    global contractor_service
    if not contractor_service:
        raise HTTPException(status_code=400, detail="Project not initialized.")
        
    try:
        # Pass background_tasks to service
        triggered = await contractor_service.submit_grade(
            task_id, 
            request.is_relevant, 
            request.contractor_id,
            background_tasks=background_tasks
        )
        return {"status": "submitted", "loop_triggered": triggered}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/contractor/batch/submit")
async def submit_contractor_batch(request: ContractorBatchSubmitRequest, background_tasks: BackgroundTasks):
    global contractor_service
    if not contractor_service:
        raise HTTPException(status_code=400, detail="Project not initialized.")
        
    try:
        triggered = await contractor_service.submit_batch_grades(
            request.submissions, 
            request.contractor_id,
            background_tasks=background_tasks
        )
        return {"status": "submitted", "loop_triggered": triggered}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/rubrics", response_model=List[RubricResponse])
async def get_rubrics():
    """Get all rubrics for the project."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    
    project_id = orchestrator.project.id
    storage = orchestrator.storage
    
    async with storage.session() as s:
        from storage.repositories import RubricRepository
        from storage.models import RubricModel, RubricPrincipleModel
        from sqlalchemy import select
        
        stmt = select(RubricModel).where(RubricModel.project_id == project_id).order_by(RubricModel.iteration)
        result = await s.execute(stmt)
        rubrics = result.scalars().all()
        
        response = []
        for r in rubrics:
            stmt_p = select(RubricPrincipleModel).where(RubricPrincipleModel.rubric_id == r.id)
            res_p = await s.execute(stmt_p)
            principles = res_p.scalars().all()
            
            response.append(RubricResponse(
                id=r.id,
                iteration=r.iteration,
                principles=[
                    RubricPrincipleResponse(id=p.id, description=p.description, weight=p.weight)
                    for p in principles
                ]
            ))
        return response

@app.get("/rubrics/latest/scores", response_model=List[NodeScoreResponse])
async def get_latest_rubric_scores():
    """Get scores for all nodes for the latest rubric iteration."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    
    iteration = len(orchestrator.project.rubrics)
    if iteration == 0:
        return []
        
    return await get_rubric_scores(iteration)
@app.get("/rubrics/{iteration}/scores", response_model=List[NodeScoreResponse])
async def get_rubric_scores(iteration: int):
    """Get scores for all nodes for a specific rubric iteration."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    
    project_id = orchestrator.project.id
    storage = orchestrator.storage
    
    async with storage.session() as s:
        from storage.models import NodeModel, PrincipleScoreModel, RubricModel, RubricPrincipleModel
        from sqlalchemy import select
        
        # 1. Get all principles for this iteration
        stmt_r = select(RubricModel).where(
            RubricModel.project_id == project_id,
            RubricModel.iteration == iteration
        )
        res_r = await s.execute(stmt_r)
        # Use first() to be robust against potential duplicates from previous bugs
        rubric = res_r.scalars().first()
        if not rubric:
            raise HTTPException(status_code=404, detail="Rubric not found")
            
        stmt_p = select(RubricPrincipleModel).where(RubricPrincipleModel.rubric_id == rubric.id)
        res_p = await s.execute(stmt_p)
        principles = res_p.scalars().all()
        principle_ids = [p.id for p in principles]
        
        # 2. Get all nodes for the project
        stmt_n = select(NodeModel).where(NodeModel.project_id == project_id)
        res_n = await s.execute(stmt_n)
        nodes = res_n.scalars().all()
        
        # 3. Get all scores for these principles and nodes
        stmt_s = select(PrincipleScoreModel).where(
            PrincipleScoreModel.principle_id.in_(principle_ids),
            PrincipleScoreModel.node_id.in_([n.id for n in nodes])
        )
        res_s = await s.execute(stmt_s)
        scores = res_s.scalars().all()
        
        # Group scores by node_id
        node_scores_map = {}
        for score in scores:
            if score.node_id not in node_scores_map:
                node_scores_map[score.node_id] = []
            node_scores_map[score.node_id].append(NodePrincipleScoreResponse(
                principle_id=score.principle_id,
                score=score.score,
                reasoning=score.reasoning
            ))
            
        # 4. Build response
        # Pre-calculate full path content for all nodes
        node_id_map = {n.id: n for n in nodes}
        full_path_map = {}
        
        def get_path(node_id):
            if node_id in full_path_map:
                return full_path_map[node_id]
            
            n = node_id_map.get(node_id)
            if not n or n.depth == 0:
                return ""
            
            parent_path = get_path(n.parent_id) if n.parent_id else ""
            path = f"{parent_path} {n.content}".strip()
            full_path_map[node_id] = path
            return path

        response = []
        top_k = orchestrator.rubric_scorer.top_k
        
        # Create a weight map for principles
        principle_weight_map = {p.id: p.weight for p in principles}
        
        for n in nodes:
            if n.depth == 0:
                continue
            principle_scores = node_scores_map.get(n.id, [])
            if not principle_scores:
                continue
            
            # Calculate aggregate score for this specific rubric using Top-K logic
            if principle_scores:
                # Sort by score descending
                sorted_ps = sorted(principle_scores, key=lambda x: x.score, reverse=True)
                # Take top K
                top_k_ps = sorted_ps[:min(top_k, len(sorted_ps))]
                # Weighted average
                total_weighted_score = sum(ps.score * principle_weight_map.get(ps.principle_id, 1.0) for ps in top_k_ps)
                total_weight = sum(principle_weight_map.get(ps.principle_id, 1.0) for ps in top_k_ps)
                agg_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            else:
                agg_score = 0.0
                
            response.append(NodeScoreResponse(
                node_id=n.id,
                content=n.content,
                full_path_content=get_path(n.id),
                aggregate_score=agg_score,
                principle_scores=principle_scores
            ))
            
        return response


@app.get("/api/rubrics/metrics")
async def get_rubric_metrics():
    """Get rubric effectiveness metrics over time."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Project not initialized.")
    if not orchestrator.storage:
        return []
    
    return await orchestrator.storage.get_rubric_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
