"""
Provides relevance feedback on nodes, simulating human input.
"""
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import random
import logging
import json

from data_models import TaxonomyNode
from llm_client import LLMClient
from template_utils import render_template_with_prompts

logger = logging.getLogger(__name__)


class FeedbackProvider(ABC):
    """Abstract interface for providing feedback on nodes."""
    
    @abstractmethod
    async def get_feedback(
        self,
        node: TaxonomyNode,
        product_description: str
    ) -> Tuple[bool, str]:
        """Get relevance feedback for a single node."""
        pass

    async def get_feedback_batch(
        self,
        nodes: List[TaxonomyNode],
        product_description: str
    ) -> List[Tuple[bool, str]]:
        """
        Get relevance feedback for multiple nodes in batch.
        Default implementation just loops get_feedback.
        """
        results = []
        for node in nodes:
            results.append(await self.get_feedback(node, product_description))
        return results


class RandomFeedbackProvider(FeedbackProvider):
    """Randomly accepts nodes based on a probability."""
    
    def __init__(self, acceptance_rate: float = 0.5):
        self.acceptance_rate = acceptance_rate
        
    async def get_feedback(self, node: TaxonomyNode, product_description: str) -> Tuple[bool, str]:
        # Simulate delay
        await asyncio.sleep(0.1)
        
        is_relevant = random.random() < self.acceptance_rate
        reasoning = "Random simulation"
        logger.info(f"Random feedback for '{node.content[:30]}...': {is_relevant}")
        return is_relevant, reasoning


class LLMFeedbackProvider(FeedbackProvider):
    """Uses an LLM to simulate a human product manager evaluating relevance."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    async def get_feedback(self, node: TaxonomyNode, product_description: str) -> Tuple[bool, str]:
        system_prompt, user_prompt = render_template_with_prompts(
            "feedback/check_relevance.j2",
            node=node,
            product_description=product_description,
            full_path_content=node.get_full_path_content()
        )
        
        def validate_feedback(data):
            if "is_relevant" not in data:
                raise ValueError("Missing 'is_relevant' field")
            if not isinstance(data["is_relevant"], bool):
                raise ValueError("'is_relevant' must be boolean")
                
        try:
            logger.info(f"Asking LLM for feedback on '{node.content[:30]}...'")
            data = await self.llm_client.complete_structured(
                user_prompt,
                system_prompt=system_prompt,
                validator=validate_feedback
            )
            
            is_relevant = data.get("is_relevant", False)
            reasoning = data.get("reasoning", "No reasoning provided")
            
            logger.info(f"LLM feedback: {is_relevant} ({reasoning[:50]}...)")
            return is_relevant, reasoning
            
        except Exception as e:
            logger.error(f"Error getting LLM feedback: {e}")
            # Fallback to False (safest option for 'relevant' check)
            return False, f"Error: {e}"

    async def get_feedback_batch(
        self,
        nodes: List[TaxonomyNode],
        product_description: str
    ) -> List[Tuple[bool, str]]:
        """Batch implementation using single LLM call."""
        if not nodes:
            return []
            
        logger.info(f"Batch feedback for {len(nodes)} nodes")
        
        system_prompt, user_prompt = render_template_with_prompts(
            "feedback/check_relevance_batch.j2",
            nodes=nodes,
            product_description=product_description
        )
        
        def validate_batch(data):
            if not isinstance(data, list):
                raise ValueError("Output must be a list")
            # Relaxed length check, but good to have
        
        try:
            data = await self.llm_client.complete_structured(
                user_prompt,
                system_prompt=system_prompt,
                validator=validate_batch
            )
            
            # Map results by index
            results_map = {}
            for item in data:
                idx = item.get("node_index")
                if idx is not None:
                    results_map[int(idx)] = (
                        bool(item.get("is_relevant", False)),
                        item.get("reasoning", "No reasoning provided")
                    )
            
            # Reconstruct list
            results = []
            for i in range(len(nodes)):
                if i in results_map:
                    results.append(results_map[i])
                else:
                    logger.warning(f"Batch feedback missed node index {i}, defaulting to False")
                    results.append((False, "Missed by LLM batch"))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch feedback: {e}")
            # Fallback to individual loop if batch fails? Or just fail safe
            # Let's fallback to False
            return [(False, f"Batch Error: {e}") for _ in nodes]
