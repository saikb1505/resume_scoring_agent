# database.py
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Example MySQL connection string
# Format: mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME
DATABASE_URL = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ResumeScore(Base):
    __tablename__ = "resume_scores"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    job_description = Column(String(1000))
    score = Column(Integer)
    strengths = Column(JSON)
    weaknesses = Column(JSON)
    fit = Column(String(50))

# Create tables (only run once at startup)
Base.metadata.create_all(bind=engine)
