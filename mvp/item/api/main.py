from typing import List
import os

import psycopg
from fastapi import FastAPI, Query
from pydantic import BaseModel


class Item(BaseModel):
    item_no: str
    item_desc: str
    item_status: str
    brand_name: str
    category: str
    class_: str
    sub_class: str
    country: str


app = FastAPI(title="Item MVP API")


def get_conn():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "demand_mvp"),
        user=os.getenv("POSTGRES_USER", "demand"),
        password=os.getenv("POSTGRES_PASSWORD", "demand"),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/items", response_model=List[Item])
def list_items(limit: int = Query(default=50, le=1000)):
    sql = """
      SELECT item_no, item_desc, item_status, brand_name, category, class, sub_class, country
      FROM dim_item
      ORDER BY item_no
      LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()

    return [
        Item(
            item_no=r[0],
            item_desc=r[1],
            item_status=r[2],
            brand_name=r[3],
            category=r[4],
            class_=r[5],
            sub_class=r[6],
            country=r[7],
        )
        for r in rows
    ]
