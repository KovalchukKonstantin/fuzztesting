# Async Storage Layer

Async PostgreSQL storage layer using SQLAlchemy with efficient batching and caching.

## Features

- **Async/await throughout** - Non-blocking database operations
- **Batch operations** - Queues updates and flushes in batches (100 items default)
- **Efficient queries** - Uses indexes and recursive CTEs for tree operations
- **Caching** - Principle scores cached to avoid recomputation
- **Transaction safety** - All operations wrapped in transactions

## Quick Start

```python
from storage import Database, AsyncStorageAdapter

# Initialize
db = Database("postgresql+asyncpg://user:pass@host/dbname")
await db.create_tables()

storage = AsyncStorageAdapter(db, project_id="my_project")

# Save project
await storage.save_project(project)

# Load project
project = await storage.load_project()

# Batch operations (auto-flush at 100 items)
await storage.save_nodes_batch(new_nodes)
await storage.update_scores_batch(node_scores)
await storage.flush()  # Manual flush if needed
```

## Architecture

### Models (`models.py`)
SQLAlchemy ORM models with relationships:
- `ProjectModel` - Projects
- `NodeModel` - Taxonomy tree nodes
- `RubricModel` - Rubrics
- `RubricPrincipleModel` - Principles
- `LabeledSampleModel` - Human feedback
- `PrincipleScoreModel` - Score cache
- `BranchStatsModel` - Branch statistics cache

### Repositories (`repositories.py`)
Convert between dataclasses and ORM:
- `NodeRepository` - Node operations with batching
- `ProjectRepository` - Project load/save
- `RubricRepository` - Rubric operations
- `LabeledSampleRepository` - Human feedback
- `PrincipleScoreRepository` - Score caching
- `BranchStatsRepository` - Statistics caching

### Adapter (`adapter.py`)
High-level interface with batching queues:
- Queues operations until batch_size reached
- Auto-flushes on batch_size or manual flush
- Transaction management

## Batch Operations

All operations support batching:

```python
# Nodes - 100 nodes saved in 1 query
await storage.save_nodes_batch(nodes)

# Scores - 100 updates in 1 query
await storage.update_scores_batch([
    (node_id, rubric_score, ucb_score),
    ...
])

# Statuses - 100 updates in 1 query
await storage.update_status_batch([
    (node_id, status),
    ...
])
```

## Performance

- **Batch inserts**: 1 query for 100 nodes (vs 100 queries)
- **Batch updates**: 1 query for 100 scores (vs 100 queries)
- **Recursive CTEs**: Efficient subtree operations
- **Indexed queries**: Fast alive node lookups

## Dependencies

```bash
pip install sqlalchemy[asyncio] asyncpg
```

## Database URL Format

```
postgresql+asyncpg://user:password@host:port/database
```
