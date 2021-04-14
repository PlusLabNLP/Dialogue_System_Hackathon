from sqlalchemy import Column, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import LONGTEXT

Base = declarative_base()

class dialog(Base):
    __tablename__ = 'dialog_log'
    user_id = Column(Text(), primary_key=True)
    log = Column(LONGTEXT())