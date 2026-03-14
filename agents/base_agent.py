from pydantic import BaseModel


class BaseDevOpsAgent:
    """
    Base class for all DevOps AI agents in this project.

    Provides a common interface and avoids the incorrect pydantic_ai.Agent
    inheritance used in earlier versions.  Subclasses should:

    1. Define a ``Config`` inner class (or accept a Pydantic model via
       ``__init__``).
    2. Implement ``run()`` which returns the agent's main result.
    """

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")
