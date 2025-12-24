"""
Core data models for the test suite generation system.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class NodeStatus(Enum):
    """Status of a taxonomy node."""
    ALIVE = "alive"
    KILLED = "killed"
    HUMAN_VERIFIED_RELEVANT = "human_verified_relevant"


@dataclass
class TaxonomyNode:
    """Represents a node in the taxonomy tree."""
    id: str
    content: str  # The test scenario description
    children: List['TaxonomyNode'] = field(default_factory=list)
    parent: Optional['TaxonomyNode'] = None
    depth: int = 0
    created_iteration: int = 0  # The iteration number when this node was created
    
    # Status
    status: NodeStatus = NodeStatus.ALIVE
    human_label: Optional[bool] = None  # True=relevant, False=irrelevant, None=unlabeled
    
    # Scoring
    rubric_score: float = 0.0
    visit_count: int = 0
    
    # Computed (cached)
    ucb_score: float = 0.0
    
    def is_alive(self) -> bool:
        """Check if node is alive (not killed)."""
        return self.status != NodeStatus.KILLED
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0
        
    def get_full_path_content(self, separator: str = " ") -> str:
        """
        Build the full scenario by concatenating content from all nodes
        along the path from the root's children to this node.
        """
        parts = []
        curr = self
        while curr and curr.depth > 0:
            parts.append(curr.content)
            curr = curr.parent
        
        # Reverse to get top-to-bottom order
        return separator.join(reversed(parts))


@dataclass
class RubricPrinciple:
    """A single principle in a rubric."""
    description: str  # Natural language description
    weight: float = 1.0  # Importance weight (0.0-1.0)
    id: Optional[str] = None  # Database ID (set when loaded from storage)


@dataclass
class Rubric:
    """A rubric containing multiple principles."""
    principles: List[RubricPrinciple] = field(default_factory=list)
    id: Optional[str] = None
    
    def add_principle(self, principle: RubricPrinciple):
        """Add a new principle to the rubric."""
        self.principles.append(principle)
    
    def __len__(self) -> int:
        return len(self.principles)


@dataclass
class PrincipleScore:
    """Score for a single principle on a node."""
    principle: RubricPrinciple
    score: float  # 0-10
    weight: float
    reasoning: Optional[str] = None  # LLM's explanation


@dataclass
class BranchStats:
    """Statistics for a top-level branch."""
    root_child: TaxonomyNode  # Which top-level branch
    total_visits: int = 0
    total_nodes: int = 0
    killed_count: int = 0
    verified_relevant_count: int = 0


@dataclass
class RubricMetrics:
    """Effectiveness metrics for a specific rubric iteration."""
    num_principles: int
    new_principles_count: int
    merged_principles_count: int
    avg_score: float
    score_variance: float
    score_alignment: Optional[float] = None  # (avg_relevant_score - avg_irrelevant_score)


@dataclass
class LabeledSample:
    """A node that has been labeled by a human."""
    node: TaxonomyNode
    label: bool  # True=relevant, False=irrelevant
    iteration: int  # Which iteration this was labeled in


@dataclass
class Project:
    """Represents a test suite generation project."""
    product_description: str
    taxonomy_root: TaxonomyNode
    rubrics: List[Rubric] = field(default_factory=list)
    human_labeled_samples: List[LabeledSample] = field(default_factory=list)
    new_branch_count: int = 0  # How many new branches created
    total_expansions: int = 0  # Total number of node expansions
    
    def get_current_rubric(self) -> Optional[Rubric]:
        """Get the most recent rubric, or None if none exist."""
        if not self.rubrics:
            return None
        # Return the latest rubric (Snapshot strategy)
        return self.rubrics[-1]
