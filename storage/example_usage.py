"""
Example usage of async storage layer.
"""
import asyncio
from storage import Database, AsyncStorageAdapter
from data_models import Project, TaxonomyNode, NodeStatus


async def main():
    # Initialize database
    db = Database("postgresql+asyncpg://user:password@localhost/testdb")
    
    # Create tables
    await db.create_tables()
    
    # Create storage adapter
    storage = AsyncStorageAdapter(db, project_id="my_project")
    
    # Create a project
    root = TaxonomyNode(id="root", content="root")
    project = Project(
        product_description="An AI customer support agent",
        taxonomy_root=root
    )
    project.id = "my_project"
    
    # Save project
    await storage.save_project(project)
    print("Project saved!")
    
    # Load project
    loaded_project = await storage.load_project()
    print(f"Loaded project: {loaded_project.product_description}")
    
    # Add some nodes
    child1 = TaxonomyNode(id="child1", content="Test scenario 1", parent=root, depth=1)
    child2 = TaxonomyNode(id="child2", content="Test scenario 2", parent=root, depth=1)
    root.children = [child1, child2]
    
    # Save nodes (batched automatically)
    await storage.save_nodes_batch([child1, child2])
    await storage.flush()  # Manual flush
    
    # Update scores (batched)
    node_scores = [
        ("child1", 0.8, 1.2),
        ("child2", 0.6, 1.0)
    ]
    await storage.update_scores_batch(node_scores)
    await storage.flush()
    
    # Get alive nodes
    alive_nodes = await storage.get_alive_nodes()
    print(f"Found {len(alive_nodes)} alive nodes")
    
    # Cleanup
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
