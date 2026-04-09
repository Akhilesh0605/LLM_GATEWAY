from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession
from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy import Column,String,Integer,Float,Boolean,DateTime,text
from datetime import datetime,timezone
from app.config import get_settings
import uuid

settings=get_settings()
engine=create_async_engine(settings.POSTGRESQL_LINK)
Base=declarative_base()
async_session=sessionmaker(engine,class_=AsyncSession,expire_on_commit=False)

class RequestLog(Base):
    __tablename__="request_logs"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id       = Column(String, nullable=False)
    query            = Column(String, nullable=False)
    model_used       = Column(String, nullable=False)
    complexity_score = Column(Integer, nullable=False)
    tier             = Column(String, nullable=False)
    cache_hit        = Column(Boolean, nullable=False)
    latency_ms       = Column(Float, nullable=False)
    tokens_used      = Column(Integer, nullable=False)
    cost_usd         = Column(Float, nullable=False)
    timestamp        = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class EvaluationLog(Base):
    __tablename__="evaluation_logs"
    id =Column(String,primary_key=True,default=lambda:str(uuid.uuid4()))
    request_id=Column(String,nullable=False)
    query=Column(String,nullable=False)
    routed_model=Column(String,nullable=False)
    complexity_score=Column(Integer,nullable=False)
    agreement_score=Column(Float,nullable=False)
    triggered_by=Column(String,nullable=False)
    timestamp=Column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def log_request(
        request_id:str,
        query:str,
        model_used:str,
        complexity_score:int,
        tier:str,
        cache_hit:bool,
        latency_ms:float,
        tokens_used:int,
        cost_usd:float
)-> None :
    log_entry=RequestLog(
        request_id=request_id,
        query=query,
        model_used=model_used,
        complexity_score=complexity_score,
        tier=tier,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_usd=cost_usd
    )

    async with async_session() as session:
        async with session.begin():
            session.add(log_entry)


async def log_evaluation( 
    request_id: str,
    query: str,
    routed_model: str,
    complexity_score: int,
    agreement_score: float,
    triggered_by: str
    )-> None:
        evaluation_log=EvaluationLog(
            request_id=request_id,
            query= query,
            routed_model=routed_model,
            complexity_score=complexity_score,
            agreement_score=agreement_score,
            triggered_by=triggered_by
        )

        async with async_session() as session:
            async with session.begin():
                session.add(evaluation_log)



async def get_analytics() -> dict:
    sql_query=text("""
        SELECT
            COUNT(*) as total_requests,
            AVG(latency_ms) as avg_latency_ms,
            SUM(cost_usd) as total_cost_usd,
            SUM(CASE WHEN cache_hit = true THEN 1 ELSE 0 END) as cache_hits,
            SUM(tokens_used) as total_tokens
        FROM request_logs """
    )

    async with async_session() as session:
        result = await session.execute(sql_query)
        row=result.fetchone()

    total_requests = row[0] or 0
    avg_latency_ms = row[1] or 0.0
    total_cost_usd = row[2] or 0.0
    cache_hits     = row[3] or 0
    total_tokens   = row[4] or 0

    cache_hit_rate=(cache_hits/total_requests*100) if total_requests>0 else 0.0
    baseline_cost=(total_tokens/1_000_000)*2.00
    cost_saved_usd=baseline_cost-total_cost_usd
    cost_saved_percent = (cost_saved_usd / baseline_cost * 100) if baseline_cost > 0 else 0.0

    return {
    "total_requests": total_requests,
    "avg_latency_ms": round(avg_latency_ms, 2),
    "total_cost_usd": round(total_cost_usd, 4),
    "cache_hit_rate": round(cache_hit_rate, 2),
    "cost_saved_usd": round(cost_saved_usd, 4),
    "cost_saved_percent": round(cost_saved_percent, 2),
    "total_tokens": total_tokens,
    }

async def get_benchmark()-> dict:
    model_distribution_query=text("""
        SELECT model_used,COUNT(*) as count
        FROM request_logs
        GROUP BY model_used
    """)

    # routing_accuracy=text("""
    #     SELECT AVG(agreement_score) as avg_agreement
    #     FROM evaluation_logs
    # """)

    async with async_session() as session:
        result= await session.execute(model_distribution_query)
        rows=result.fetchall()

    model_usage={}
    total=0

    for row in rows :
        model_name=row[0]
        count=row[1]
        model_usage[model_name]=count
        total+=count
    
    model_distribution={}
    for model_name,count in model_usage.items():
        percentage=(count/total*100) if total>0.0 else 0
        model_distribution[model_name]={
            "count":count,
            "percentage":round(percentage,2)
        }

    analytics = await get_analytics()
    cache_hit_rate = analytics["cache_hit_rate"]
    cost_saved_percent = analytics["cost_saved_percent"]

    return {
    "total_requests": total,
    "model_distribution": model_distribution,
    "cache_hit_rate": cache_hit_rate,
    "cost_saved_percent": cost_saved_percent,
    }