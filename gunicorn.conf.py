"""
Gunicorn configuration for Render.
"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '4000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "sync"
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
proc_name = "union-budget-rag"
