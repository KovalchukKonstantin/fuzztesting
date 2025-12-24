"""
Scores nodes using rubrics with per-principle scoring and Top-K aggregation.
"""
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import logging
import asyncio

from data_models import TaxonomyNode, Rubric, RubricPrinciple, PrincipleScore
from llm_client import LLMClient
from template_utils import render_template_with_prompts

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class LLMScorer(ABC):
    """Abstract interface for LLM-based principle scoring."""
    
    @abstractmethod
    async def score_principles_batch(
        self,
        node: TaxonomyNode,
        principles: List[RubricPrinciple],
        product_description: str
    ) -> List[Tuple[float, Optional[str]]]:
        """
        Score a node against multiple principles in a single call.
        
        Args:
            node: Node to score
            principles: List of principles to evaluate against
            product_description: Product context
        
        Returns:
            List of (score, reasoning) tuples corresponding to the input order.
        """
        pass


class RubricScorer:
    """
    Scores nodes against rubrics using per-principle scoring with Top-K aggregation.
    """
    
    def __init__(self, llm_scorer: LLMScorer, top_k: int = 2):
        """
        Args:
            llm_scorer: LLM implementation for scoring principles
            top_k: Number of top principles to average (default 2)
        """
        self.llm_scorer = llm_scorer
        self.top_k = top_k
    
    async def score_node(
        self,
        node: TaxonomyNode,
        rubric: Rubric,
        product_description: str
    ) -> Tuple[float, List[PrincipleScore]]:
        """
        Score a node against a rubric.
        
        Args:
            node: Node to score
            rubric: Rubric to score against
            product_description: Product context
        
        Returns:
            Tuple of (aggregate score, list of principle scores with details)
        """
        if not rubric.principles:
            return 0.0, []
        
        # Batch call for ALL principles
        raw_scores = await self.llm_scorer.score_principles_batch(node, rubric.principles, product_description)
        
        principle_scores = []
        for i, (score, reasoning) in enumerate(raw_scores):
            principle = rubric.principles[i]
            principle_scores.append(PrincipleScore(
                principle=principle,
                score=score,
                weight=principle.weight,
                reasoning=reasoning
            ))
        
        # Top-K aggregation
        sorted_scores = sorted(
            principle_scores,
            key=lambda ps: ps.score,
            reverse=True
        )
        
        top_k_scores = sorted_scores[:min(self.top_k, len(sorted_scores))]
        
        # Weighted average of top K
        total = sum(ps.score * ps.weight for ps in top_k_scores)
        weight_sum = sum(ps.weight for ps in top_k_scores)
        
        aggregate_score = total / weight_sum if weight_sum > 0 else 0.0
        
        return aggregate_score, principle_scores
    
    async def score_all_alive(
        self,
        nodes: List[TaxonomyNode],
        rubric: Rubric,
        product_description: str,
        cached_scores: Optional[dict[str, dict[str, Tuple[float, Optional[str]]]]] = None
    ) -> dict:
        """
        Score all nodes against rubric.
        
        Only scores nodes against principles they haven't been scored against before.
        Uses cached scores for principles that were already evaluated.
        
        Args:
            nodes: List of nodes to score
            rubric: Rubric to score against
            product_description: Product context
            cached_scores: Optional dict mapping node_id -> principle_id -> (score, reasoning)
        
        Returns:
            Dictionary mapping node_id to (score, principle_scores)
        """
        if cached_scores is None:
            cached_scores = {}
        
        # Create tasks for all nodes
        tasks = []
        node_id_list = []
        
        for node in nodes:
            node_id_list.append(node.id)
            node_cached = cached_scores.get(node.id, {})
            tasks.append(self._score_node_with_cache(node, rubric, product_description, node_cached))
            
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Assemble results
        results = {}
        for node_id, (agg_score, p_scores) in zip(node_id_list, results_list):
            results[node_id] = (agg_score, p_scores)
            
        return results

    async def _score_node_with_cache(
        self,
        node: TaxonomyNode,
        rubric: Rubric,
        product_description: str,
        node_cached: dict
    ) -> Tuple[float, List[PrincipleScore]]:
        """Helper to score a single node using cache where possible."""
        
        principles_to_score = []
        indices_to_score = []
        final_scores: List[Optional[PrincipleScore]] = [None] * len(rubric.principles)
        
        # 1. Check cache and identify missing principles
        for i, principle in enumerate(rubric.principles):
            principle_id = principle.id if principle.id else None
            
            if principle_id and principle_id in node_cached:
                # Cache Hit
                cached_score, cached_reasoning = node_cached[principle_id]
                final_scores[i] = PrincipleScore(
                    principle=principle,
                    score=cached_score,
                    weight=principle.weight,
                    reasoning=cached_reasoning
                )
            else:
                # Cache Miss - Needs Scoring
                principles_to_score.append(principle)
                indices_to_score.append(i)
        
        # 2. Batch score missing principles (if any)
        if principles_to_score:
            # product_description passed as arg now
            raw_scores = await self.llm_scorer.score_principles_batch(node, principles_to_score, product_description)
            
            # Map back to final_scores
            for j, (score, reasoning) in enumerate(raw_scores):
                original_index = indices_to_score[j]
                principle = rubric.principles[original_index]
                final_scores[original_index] = PrincipleScore(
                    principle=principle,
                    score=score,
                    weight=principle.weight,
                    reasoning=reasoning
                )
        
        # Ensure all are filled (should be)
        principle_scores = [ps for ps in final_scores if ps is not None]

        # Top-K aggregation
        sorted_scores = sorted(
            principle_scores,
            key=lambda ps: ps.score,
            reverse=True
        )
        
        top_k_scores = sorted_scores[:min(self.top_k, len(sorted_scores))]
        
        # Weighted average of top K
        total = sum(ps.score * ps.weight for ps in top_k_scores)
        weight_sum = sum(ps.weight for ps in top_k_scores)
        
        aggregate_score = total / weight_sum if weight_sum > 0 else 0.0
        node.rubric_score = aggregate_score
        
        return aggregate_score, principle_scores


class RealLLMScorer(LLMScorer):
    """Real implementation using LLM Client and Jinja2 templates."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    async def score_principle(self, node: TaxonomyNode, principle: RubricPrinciple) -> Tuple[float, Optional[str]]:
        # Keep for backward compatibility or single calls
        return await self._score_single_internal(node, principle)

    async def _score_single_internal(self, node: TaxonomyNode, principle: RubricPrinciple) -> Tuple[float, Optional[str]]:
         logger.info(f"Scoring node {node.content[:30]}... against principle {principle.description[:30]}...")
         system_prompt, user_prompt = render_template_with_prompts(
             "scoring/score_principle.j2",
             node=node,
             principle=principle,
             full_path_content=node.get_full_path_content()
         )
         try:
              # Expecting {"score": 0 or 1, "reasoning": str}
              def validate_score(data):
                  if "score" not in data:
                      raise ValueError("Missing 'score' field")
                  if not isinstance(data["score"], (int, float)):
                      raise ValueError("'score' must be a number")
                  if not (0 <= float(data["score"]) <= 10):
                      raise ValueError("'score' must be between 0 and 10")
                      
              data = await self.llm_client.complete_structured(
                  user_prompt,
                  system_prompt=system_prompt,
                  validator=validate_score
              )
              return float(data.get("score", 0.0)), data.get("reasoning", "")
         except Exception as e:
              logger.error(f"Error scoring principle: {e}")
              return 0.0, f"Error: {e}"

    async def score_principles_batch(
        self,
        node: TaxonomyNode,
        principles: List[RubricPrinciple],
        product_description: str
    ) -> List[Tuple[float, Optional[str]]]:
        logger.info(f"Batch scoring node {node.content[:30]}... against {len(principles)} principles")
        
        system_prompt, user_prompt = render_template_with_prompts(
            "scoring/score_batch.j2",
            node=node,
            principles=principles,
            full_path_content=node.get_full_path_content(),
            product_description=product_description
        )
        
        def validate_batch(data):
            if not isinstance(data, list):
                raise ValueError("Output must be a list")
            if len(data) != len(principles):
                # We can relax this if LLM misses one, but ideally strict
                # For robustness, we will map by index if provided
                pass 
                
        try:
            data = await self.llm_client.complete_structured(
                user_prompt,
                system_prompt=system_prompt,
                validator=validate_batch
            )
            
            # Map results by index
            # Schema: [{"principle_index": 0, "score": X, ...}]
            
            results_map = {}
            for item in data:
                idx = item.get("principle_index")
                if idx is not None:
                    results_map[int(idx)] = (float(item.get("score", 0.0)), item.get("reasoning", ""))
            
            # Reconstruct list from map in order
            ordered_results = []
            for i in range(len(principles)):
                if i in results_map:
                    ordered_results.append(results_map[i])
                else:
                    logger.warning(f"Batch scoring missed principle index {i}, defaulting to 0")
                    ordered_results.append((0.0, "Missed by LLM batch"))
            
            return ordered_results
            
        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            # Fallback to 0s
            return [(0.0, f"Error: {e}") for _ in principles]
