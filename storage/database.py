"""
Database setup and initialization for async PostgreSQL.
"""
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from storage.models import Base


class Database:
    """Database connection manager."""
    
    def __init__(self, database_url: str):
        """
        Initialize database.
        
        Args:
            database_url: PostgreSQL async URL (e.g., "postgresql+asyncpg://user:pass@host/db")
        """
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL logging
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self):
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all tables (for testing)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @asynccontextmanager
    async def get_session(self):
        """Get async session as context manager."""
        async with self.async_session_maker() as session:
            yield session
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
