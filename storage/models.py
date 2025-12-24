"""
SQLAlchemy ORM models for async PostgreSQL.
"""
from sqlalchemy import String, Integer, Float, Boolean, Text, ForeignKey, TIMESTAMP, UniqueConstraint, Index
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ProjectModel(Base):
    """Project ORM model."""
    __tablename__ = 'projects'
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default='default')
    product_description: Mapped[str] = mapped_column(Text, nullable=False)
    new_branch_count: Mapped[int] = mapped_column(Integer, default=0)
    total_expansions: Mapped[int] = mapped_column(Integer, default=0)
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    updated_at = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    nodes: Mapped[list["NodeModel"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    rubrics: Mapped[list["RubricModel"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    labeled_samples: Mapped[list["LabeledSampleModel"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    branch_stats: Mapped[list["BranchStatsModel"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class NodeModel(Base):
    """Node ORM model (taxonomy tree)."""
    __tablename__ = 'nodes'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String, ForeignKey('nodes.id', ondelete='CASCADE'), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Status
    status: Mapped[str] = mapped_column(String, nullable=False, default='alive')
    human_label: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    
    # Scoring
    rubric_score: Mapped[float] = mapped_column(Float, default=0.0)
    visit_count: Mapped[int] = mapped_column(Integer, default=0)
    ucb_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Metadata
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    updated_at = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project: Mapped["ProjectModel"] = relationship(back_populates="nodes")
    parent: Mapped["NodeModel | None"] = relationship(remote_side=[id], backref="children")
    principle_scores: Mapped[list["PrincipleScoreModel"]] = relationship(back_populates="node", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_nodes_project_status', 'project_id', 'status'),  # For get_alive_nodes queries
        Index('idx_nodes_parent', 'parent_id'),  # For tree traversal
    )


class RubricModel(Base):
    """Rubric ORM model."""
    __tablename__ = 'rubrics'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    project: Mapped["ProjectModel"] = relationship(back_populates="rubrics")
    principles: Mapped[list["RubricPrincipleModel"]] = relationship(back_populates="rubric", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_rubrics_project_iteration', 'project_id', 'iteration'),  # For getting rubrics by project
    )


class RubricMetricsModel(Base):
    """Rubric effectiveness metrics."""
    __tablename__ = 'rubric_metrics'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    rubric_id: Mapped[str] = mapped_column(String, ForeignKey('rubrics.id', ondelete='CASCADE'), nullable=False)
    
    num_principles: Mapped[int] = mapped_column(Integer, nullable=False)
    new_principles_count: Mapped[int] = mapped_column(Integer, nullable=False)
    merged_principles_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    avg_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_variance: Mapped[float] = mapped_column(Float, nullable=False)
    score_alignment: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    rubric: Mapped["RubricModel"] = relationship(backref="metrics")


class RubricPrincipleModel(Base):
    """Rubric principle ORM model."""
    __tablename__ = 'rubric_principles'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    rubric_id: Mapped[str] = mapped_column(String, ForeignKey('rubrics.id', ondelete='CASCADE'), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    rubric: Mapped["RubricModel"] = relationship(back_populates="principles")
    principle_scores: Mapped[list["PrincipleScoreModel"]] = relationship(back_populates="principle")
    
    # Note: No index on 'description' - it's a large Text field from LLM outputs.
    # Principle lookups by description are infrequent and can use sequential scan.


class LabeledSampleModel(Base):
    """Labeled sample ORM model."""
    __tablename__ = 'labeled_samples'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    node_id: Mapped[str] = mapped_column(String, ForeignKey('nodes.id', ondelete='CASCADE'), nullable=False)
    label: Mapped[bool] = mapped_column(Boolean, nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    project: Mapped["ProjectModel"] = relationship(back_populates="labeled_samples")
    node: Mapped["NodeModel"] = relationship()
    
    __table_args__ = (UniqueConstraint('node_id', name='uq_labeled_samples_node'),)


class PrincipleScoreModel(Base):
    """Principle score cache ORM model."""
    __tablename__ = 'principle_scores'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    node_id: Mapped[str] = mapped_column(String, ForeignKey('nodes.id', ondelete='CASCADE'), nullable=False)
    principle_id: Mapped[str] = mapped_column(String, ForeignKey('rubric_principles.id', ondelete='CASCADE'), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    rubric_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    updated_at = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    node: Mapped["NodeModel"] = relationship(back_populates="principle_scores")
    principle: Mapped["RubricPrincipleModel"] = relationship(back_populates="principle_scores")
    
    __table_args__ = (
        UniqueConstraint('node_id', 'principle_id', 'rubric_version', name='uq_principle_scores'),
        Index('idx_principle_scores_node_version', 'node_id', 'rubric_version'),  # For cache lookups
    )


class BranchStatsModel(Base):
    """Branch statistics cache ORM model."""
    __tablename__ = 'branch_stats'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    branch_node_id: Mapped[str] = mapped_column(String, ForeignKey('nodes.id', ondelete='CASCADE'), nullable=False)
    total_visits: Mapped[int] = mapped_column(Integer, default=0)
    total_nodes: Mapped[int] = mapped_column(Integer, default=0)
    killed_count: Mapped[int] = mapped_column(Integer, default=0)
    verified_relevant_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project: Mapped["ProjectModel"] = relationship(back_populates="branch_stats")
    branch_node: Mapped["NodeModel"] = relationship()
    
    __table_args__ = (UniqueConstraint('project_id', 'branch_node_id', name='uq_branch_stats'),)


class GradingQueueItem(Base):
    """Queue items for contractor grading."""
    __tablename__ = 'grading_queue'
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    node_id: Mapped[str] = mapped_column(String, ForeignKey('nodes.id', ondelete='CASCADE'), nullable=False)
    
    # Status: 'PENDING', 'ASSIGNED'
    status: Mapped[str] = mapped_column(String, nullable=False, default='PENDING')
    assigned_to: Mapped[str | None] = mapped_column(String, nullable=True)
    
    created_at = mapped_column(TIMESTAMP, server_default=func.now())
    assigned_at = mapped_column(TIMESTAMP, nullable=True)
    
    # Relationships
    project: Mapped["ProjectModel"] = relationship()
    node: Mapped["NodeModel"] = relationship()
    
    __table_args__ = (
        Index('idx_grading_queue_status_assigned', 'status', 'assigned_at'), # For timeouts
        Index('idx_grading_queue_project_status', 'project_id', 'status'), # For fetching pending
    )
