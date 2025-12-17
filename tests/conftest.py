import pytest
import pytest_asyncio
from elysia.api.dependencies.common import get_user_manager


@pytest_asyncio.fixture(scope="session", autouse=True)
async def cleanup_clients():
    yield
    user_manager = get_user_manager()
    await user_manager.close_all_clients()
