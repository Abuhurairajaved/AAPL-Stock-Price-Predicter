from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from models import Base, StockPrediction  # Ensure this imports your Base model and StockPrediction model correctly

# Async SQLite database URL (adjust as needed)
DATABASE_URL = "sqlite+aiosqlite:///./stock_predicter.db"  # You can change this path if necessary

# Create the async database engine
engine = create_async_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# AsyncSessionLocal will be used to create session instances for database interactions
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Initialize the async database connection (remove the `databases` package)
async def init_db():
    # Ensure models are imported before this function is called
    async with engine.begin() as conn:
        # Create all tables defined in your Base model (ensure models are imported before this)
        await conn.run_sync(Base.metadata.create_all)

# Dependency to get the async session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Shutdown handler to disconnect the async database connection
async def shutdown_db():
    # Close the engine (not needed to disconnect databases package anymore)
    await engine.dispose()
    print("Database disconnected!")

# Helper function to get a prediction by ID
async def get_prediction_by_id(db: AsyncSession, prediction_id: int):
    async with db.begin():
        result = await db.execute(
            select(StockPrediction).filter(StockPrediction.id == prediction_id)
        )
        prediction = result.scalars().first()  # Returns the first result or None if not found
    return prediction
