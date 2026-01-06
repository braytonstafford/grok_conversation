"""The exceptions used by Grok Conversation."""
from homeassistant.exceptions import HomeAssistantError


class EntityNotFound(HomeAssistantError):
    """When referenced entity not found."""

    def __init__(self, entity_id: str) -> None:
        """Initialize error."""
        super().__init__(f"Unable to find entity {entity_id}")
        self.entity_id = entity_id


class EntityNotExposed(HomeAssistantError):
    """When referenced entity not exposed."""

    def __init__(self, entity_id: str) -> None:
        """Initialize error."""
        super().__init__(f"entity {entity_id} is not exposed")
        self.entity_id = entity_id


class CallServiceError(HomeAssistantError):
    """Error during service calling"""

    def __init__(self, domain: str, service: str, data: object) -> None:
        """Initialize error."""
        super().__init__(
            f"unable to call service {domain}.{service} with data {data}. One of 'entity_id', 'area_id', or 'device_id' is required"
        )
        self.domain = domain
        self.service = service
        self.data = data


class FunctionNotFound(HomeAssistantError):
    """When referenced function not found."""

    def __init__(self, function: str) -> None:
        """Initialize error."""
        super().__init__(f"function '{function}' does not exist")
        self.function = function


class NativeNotFound(HomeAssistantError):
    """When native function not found."""

    def __init__(self, name: str) -> None:
        """Initialize error."""
        super().__init__(f"native function '{name}' does not exist")
        self.name = name


class FunctionLoadFailed(HomeAssistantError):
    """When function load failed."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__(
            "failed to load functions. Verify functions are valid in a yaml format"
        )


class ParseArgumentsFailed(HomeAssistantError):
    """When parse arguments failed."""

    def __init__(self, arguments: str) -> None:
        """Initialize error."""
        super().__init__(
            f"failed to parse arguments `{arguments}`. Increase maximum token to avoid the issue."
        )
        self.arguments = arguments


class TokenLengthExceededError(HomeAssistantError):
    """When openai return 'length' as 'finish_reason'."""

    def __init__(self, token: int) -> None:
        """Initialize error."""
        super().__init__(
            f"token length(`{token}`) exceeded. Increase maximum token to avoid the issue."
        )
        self.token = token


class InvalidFunction(HomeAssistantError):
    """When function validation failed."""

    def __init__(self, function_name: str) -> None:
        """Initialize error."""
        super().__init__(
            f"failed to validate function `{function_name}`"
        )
        self.function_name = function_name