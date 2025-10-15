import nest_asyncio
nest_asyncio.apply()

from src.train import run
run('configs/config.yaml')