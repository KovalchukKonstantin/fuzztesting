"""
Derives rubrics from human feedback using accumulate strategy.
"""
from typing import List
from abc import ABC, abstractmethod
import logging

from data_models import Rubric, RubricPrinciple, LabeledSample, Project
from llm_client import LLMClient
from template_utils import render_template_with_prompts

logger = logging.getLogger(__name__)


class LLMRubricDeriver(ABC):
    """Abstract interface for LLM-based rubric derivation."""
    
    @abstractmethod
    async def derive_new_principles(
        self,
        relevant_samples: List[LabeledSample],
        irrelevant_samples: List[LabeledSample],
        current_rubric: Rubric,
        product_description: str
    ) -> List[RubricPrinciple]:
        """
        Derive new rubric principles from human feedback.
        
        Args:
            relevant_samples: Nodes labeled as relevant by humans
            irrelevant_samples: Nodes labeled as irrelevant by humans
            current_rubric: Current rubric (may be empty)
            product_description: Product description for context
        
        Returns:
            List of new principles to add
        """
        pass


class RubricDeriver:
    """
    Derives and updates rubrics from human feedback using accumulate strategy.
    """
    
    def __init__(self, llm_deriver: LLMRubricDeriver):
        """
        Args:
            llm_deriver: LLM implementation for deriving principles
        """
        self.llm_deriver = llm_deriver
    
    async def derive_rubrics(
        self,
        project: Project
    ) -> "Tuple[Rubric, int, int]":
        """
        Derive new principles and add to project's rubrics.
        
        Uses accumulate strategy: adds new principles without removing old ones.
        
        Args:
            project: Project with labeled samples and current rubrics
        
        Returns:
            Tuple: (Updated rubric, new_principles_count, merged_principles_count)
        """
        # Separate relevant and irrelevant samples
        relevant_samples = [
            sample for sample in project.human_labeled_samples
            if sample.label is True
        ]
        irrelevant_samples = [
            sample for sample in project.human_labeled_samples
            if sample.label is False
        ]
        
        # Get current rubric (the prompt will see this and validation/invalidation decisions)
        current_rubric = project.get_current_rubric() or Rubric()
        
        # Derive FULL set of principles (LLM now returns valid old + new)
        full_rubric_principles = await self.llm_deriver.derive_new_principles(
            relevant_samples=relevant_samples,
            irrelevant_samples=irrelevant_samples,
            current_rubric=current_rubric,
            product_description=project.product_description
        )
        
        if full_rubric_principles:
            # Stats for convergence tracking
            count_before = len(full_rubric_principles)
            
            # Deduplicate locally
            # NOTE: We cast to RealLLMRubricDeriver to access deduplicate_principles if available
            # Ideally this method would be on the interface or helper, but for now we check attribute
            if hasattr(self.llm_deriver, 'deduplicate_principles'):
                full_rubric_principles = self.llm_deriver.deduplicate_principles(full_rubric_principles)
            
            count_after = len(full_rubric_principles)
            merged_count = count_before - count_after
            
            logger.info(f"Rubric Convergence: Before={count_before}, After={count_after}, Merged={merged_count}")

            # "Rehydrate" IDs: Match new principles to old ones by description (fuzzy) to preserve identity
            # This is critical because the Scorer uses principle.id to reuse scores.
            old_principles_map = {p.id: p.description.lower().strip() for p in current_rubric.principles if p.id}
            old_principles_lookup = {p.id: p for p in current_rubric.principles if p.id}
            
            import difflib
            
            final_principles = []
            new_principles_count = 0
            
            for p in full_rubric_principles:
                p_desc = p.description.lower().strip()
                matched_id = None
                
                # 1. Exact match first
                for pid, old_desc in old_principles_map.items():
                    if p_desc == old_desc:
                        matched_id = pid
                        break
                
                # 2. Fuzzy match if no exact match
                if not matched_id:
                    best_ratio = 0.0
                    for pid, old_desc in old_principles_map.items():
                        ratio = difflib.SequenceMatcher(None, p_desc, old_desc).ratio()
                        if ratio > 0.8:  # 80% similarity threshold
                            if ratio > best_ratio:
                                best_ratio = ratio
                                matched_id = pid
                    
                    if matched_id and best_ratio > 0.8:
                        logger.info(f"Fuzzy matched principle: '{p.description}' -> Old ID {matched_id} (Sim: {best_ratio:.2f})")
                        
                if matched_id:
                    # Revert to the OLD principle object to ensure stability and exact identity
                    # This implies we keep the old weight/wording if fuzzy matched
                    final_principles.append(old_principles_lookup[matched_id])
                else:
                    final_principles.append(p)
                    new_principles_count += 1
            
            # Check for Convergence: If the set of IDs is identical, the rubric hasn't changed.
            # We still SAVE it to track the metric of "0 new principles" over time.
            current_ids = {p.id for p in current_rubric.principles if p.id}
            final_ids = {p.id for p in final_principles if p.id}
            
            if new_principles_count == 0 and final_ids == current_ids:
                logger.info("Rubric converged (identical to previous). Saving new version to record stability.")

            # Create new rubric with the FULL defined set
            new_rubric = Rubric(principles=final_principles)
            
            # Add to project's rubric list (the latest one is the current truth)
            project.rubrics.append(new_rubric)
            logger.info(f"Updated rubric with {len(final_principles)} principles (Strategy: Replace/Snapshot)")
            logger.info(f"Rubric Stability: {new_principles_count} new principles added this iteration")
            
            return project.get_current_rubric(), new_principles_count, merged_count
        else:
             logger.info("No principles derived (empty set)")
             return project.get_current_rubric(), 0, 0

class RealLLMRubricDeriver(LLMRubricDeriver):
    """Real implementation using LLM Client and Jinja2 templates."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    async def derive_new_principles(
        self,
        relevant_samples: List[LabeledSample],
        irrelevant_samples: List[LabeledSample],
        current_rubric: Rubric,
        product_description: str
    ) -> List[RubricPrinciple]:
        
        logger.info(f"Deriving principles from {len(relevant_samples)} relevant and {len(irrelevant_samples)} irrelevant samples")
        
        system_prompt, user_prompt = render_template_with_prompts(
            "rubric/derive_principles.j2",
            relevant_samples=relevant_samples,
            irrelevant_samples=irrelevant_samples,
            current_rubric=current_rubric,
            product_description=product_description
        )
        def validate_principles(data):
            if not isinstance(data, list):
                raise ValueError("Output must be a list of objects")
            for item in data:
                if not isinstance(item, dict):
                     raise ValueError("Items must be objects")
                if "description" not in item:
                     raise ValueError("Missing 'description' field")
                     
        try:
             data = await self.llm_client.complete_structured(
                 user_prompt,
                 system_prompt=system_prompt,
                 validator=validate_principles
             )
             # Expecting list of dicts [{"description": "...", "weight": 1.0}]
             principles = []
             for item in data:
                 principles.append(RubricPrinciple(
                     description=item.get("description", ""),
                     weight=1.0 # Force weight to 1.0
                 ))
             logger.info(f"Derived {len(principles)} new principles")
             return principles
        except Exception as e:
             logger.error(f"Error deriving principals (LLM failure): {e}", exc_info=True)
             return []

    def deduplicate_principles(self, principles: List[RubricPrinciple], threshold: float = 0.75) -> List[RubricPrinciple]:
        """
        Deduplicate principles using fuzzy string matching.
        
        Args:
            principles: List of principles to deduplicate
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            Deduplicated list of principles
        """
        import difflib
        
        unique_principles = []
        merged_count = 0
        
        # Sort by length (descending) to prefer keeping longer, more descriptive principles
        sorted_principles = sorted(principles, key=lambda p: len(p.description), reverse=True)
        
        for p in sorted_principles:
            is_duplicate = False
            for existing in unique_principles:
                similarity = difflib.SequenceMatcher(None, p.description.lower(), existing.description.lower()).ratio()
                if similarity > threshold:
                    is_duplicate = True
                    merged_count += 1
                    logger.info(f"Merging similar principles:\n  Keep: {existing.description}\n  Drop: {p.description} (Sim: {similarity:.2f})")
                    break
            
            if not is_duplicate:
                unique_principles.append(p)
                
        return unique_principles

