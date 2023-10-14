import numpy 
import pickle
import pandas as pd
import os
import time
import sqlalchemy
from sqlalchemy import create_engine,text

engine = create_engine("sqlite+pysqlite:///deephoopers.db",echo=True,future=True)
with engine.connect() as conn:
    result = conn.execute(text('PRAGMA table_info(TrackingDataTable)'))
    for col in result:
        print(col)
engine.dispose() 