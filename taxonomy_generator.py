"""
Generates taxonomy tree: initial tree and node expansion.
"""
from typing import List, Optional
from abc import ABC, abstractmethod
import logging
import uuid

from data_models import TaxonomyNode, Rubric, Project
from llm_client import LLMClient
from template_utils import render_template_with_prompts

logger = logging.getLogger(__name__)


class LLMTaxonomyGenerator(ABC):
    """Abstract interface for LLM-based taxonomy generation."""
    
    @abstractmethod
    async def generate_initial_topics(
        self,
        product_description: str,
        num_topics: int
    ) -> List[str]:
        """
        Generate initial top-level topics from product description.
        
        Args:
            product_description: Product description
            num_topics: Number of initial topics to generate
        
        Returns:
            List of topic descriptions (strings)
        """
        pass
    
    @abstractmethod
    async def generate_new_branch_topic(
        self,
        product_description: str,
        existing_branches: List[str]
    ) -> str:
        """
        Generate a new top-level branch topic.
        
        Args:
            product_description: Product description
            existing_branches: List of existing branch topics (to avoid duplicates)
        
        Returns:
            New topic description
        """
        pass
    
    @abstractmethod
    async def expand_node(
        self,
        node: TaxonomyNode,
        rubric: Optional[Rubric],
        product_description: str,
        num_children: int
    ) -> List[str]:
        """
        Generate child nodes for a given node.
        
        Args:
            node: Parent node to expand
            rubric: Current rubric (for guidance, may be None)
            product_description: Product description
            num_children: Number of children to generate
        
        Returns:
            List of child node descriptions (strings)
        """
        pass


class TaxonomyGenerator:
    """
    Generates and expands taxonomy tree.
    """
    
    def __init__(self, llm_generator: LLMTaxonomyGenerator):
        """
        Args:
            llm_generator: LLM implementation for taxonomy generation
        """
        self.llm_generator = llm_generator
    
    async def generate_initial_tree(
        self,
        product_description: str,
        num_topics: int = 5,
        current_iteration: int = 0
    ) -> TaxonomyNode:
        """
        Generate initial taxonomy tree.
        
        Args:
            product_description: Product description
            num_topics: Number of initial top-level topics
            current_iteration: Current iteration number
        
        Returns:
            Root node with initial topic branches
        """
        # Generate initial topics
        topics = await self.llm_generator.generate_initial_topics(
            product_description,
            num_topics
        )
        
        # Create root node
        root = TaxonomyNode(
            id=f"root_{uuid.uuid4().hex[:8]}", # Random ID to prevent collisions across projects
            content="root",
            depth=0,
            created_iteration=current_iteration
        )
        
        # Create child nodes
        children = []
        for i, topic in enumerate(topics):
            child = TaxonomyNode(
                id=f"topic_{uuid.uuid4().hex[:8]}", # Random ID
                content=topic,
                parent=root,
                depth=1,
                created_iteration=current_iteration
            )
            children.append(child)
        
        root.children = children
        
        return root
    
    async def create_new_branch(
        self,
        project: Project,
        current_iteration: int = 0
    ) -> TaxonomyNode:
        """
        Generate a new top-level branch.
        
        Args:
            project: Project with existing branches
            current_iteration: Current iteration number for creation metadata
        
        Returns:
            New top-level branch node
        """
        existing_branches = [
            child.content for child in project.taxonomy_root.children
        ]
        
        new_topic = await self.llm_generator.generate_new_branch_topic(
            project.product_description,
            existing_branches
        )
        
        new_node = TaxonomyNode(
            id=f"topic_{uuid.uuid4().hex[:8]}", # Random ID
            content=new_topic,
            parent=project.taxonomy_root,
            depth=1,
            created_iteration=current_iteration
        )
        
        project.taxonomy_root.children.append(new_node)
        project.new_branch_count += 1
        
        return new_node
    
    async def expand_node(
        self,
        node: TaxonomyNode,
        rubric: Optional[Rubric],
        product_description: str,
        num_children: int = 3,
        current_iteration: int = 0
    ) -> List[TaxonomyNode]:
        """
        Expand a node by generating children.
        
        Args:
            node: Node to expand
            rubric: Current rubric (for guidance)
            product_description: Product description
            num_children: Number of children to generate
        
        Returns:
            List of new child nodes
        """
        # Generate child descriptions
        child_descriptions = await self.llm_generator.expand_node(
            node,
            rubric,
            product_description,
            num_children
        )
        
        # Create child nodes
        children = []
        for description in child_descriptions:
            child = TaxonomyNode(
                id=f"node_{uuid.uuid4().hex[:8]}", # Random ID
                content=description,
                parent=node,
                depth=node.depth + 1,
                visit_count=0,
                created_iteration=current_iteration
            )
            children.append(child)
        
        # Add to parent
        node.children.extend(children)
        
        return children

class RealLLMTaxonomyGenerator(LLMTaxonomyGenerator):
    """Real implementation using LLM Client and Jinja2 templates."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    async def generate_initial_topics(self, product_description: str, num_topics: int) -> List[str]:
        logger.info(f"Generating {num_topics} initial topics")
        system_prompt, user_prompt = render_template_with_prompts(
            "taxonomy/initial_topics.j2",
            product_description=product_description,
            num_topics=num_topics
        )
        # Validator: Ensure it's a list of strings
        def validate_topics(data):
            if not isinstance(data, list):
                raise ValueError("Output must be a list of strings")
            if not all(isinstance(item, str) for item in data):
                raise ValueError("All items in list must be strings")
                
        # Use complete_structured for retries on parsing failure
        try:
            return await self.llm_client.complete_structured(
                user_prompt,
                system_prompt=system_prompt,
                validator=validate_topics
            )
        except Exception as e:
            logger.error(f"Error generating initial topics: {e}")
            return []

    async def generate_new_branch_topic(self, product_description: str, existing_branches: List[str]) -> str:
        logger.info("Generating new branch topic")
        system_prompt, user_prompt = render_template_with_prompts(
            "taxonomy/new_branch.j2",
            product_description=product_description,
            existing_branches=existing_branches
        )
        return (await self.llm_client.complete(
            user_prompt,
            system_prompt=system_prompt
        )).strip().strip('"')
    
    async def expand_node(self, node, rubric, product_description: str, num_children: int) -> List[str]:
        # Note: rubric param kept for interface compatibility but ignored
        existing_children = [child.content for child in node.children] if node.children else []
        
        logger.info(f"Expanding node: {node.content} ({num_children} children)")
        system_prompt, user_prompt = render_template_with_prompts(
            "taxonomy/expand_node.j2",
            node=node,
            product_description=product_description,
            existing_children=existing_children,
            num_children=num_children,
            full_path_content=node.get_full_path_content()
        )
        def validate_children(data):
            if not isinstance(data, list):
                raise ValueError("Output must be a list of strings")
            if not all(isinstance(item, str) for item in data):
                raise ValueError("All items in list must be strings")

        try:
            return await self.llm_client.complete_structured(
                user_prompt,
                system_prompt=system_prompt,
                validator=validate_children
            )
        except Exception as e:
            logger.error(f"Error expanding node: {e}")
            return []

