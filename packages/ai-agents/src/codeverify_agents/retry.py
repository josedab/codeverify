"""Retry utilities for resilient LLM API calls.

This module provides retry decorators with exponential backoff
for handling transient failures in LLM API calls.
"""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Type, TypeVar

import structlog

logger = structlog.get_logger()

F = TypeVar("F", bound=Callable[..., Any])


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple[Type[Exception], ...] | None = None,
    ) -> None:
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of attempts (including initial)
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exceptions to retry on.
                If None, retries on all exceptions.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )
        if self.jitter:
            # Add random jitter of up to 25% of the delay
            delay = delay * (0.75 + random.random() * 0.5)
        return delay


# Default configuration for LLM API calls
DEFAULT_LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
)


def retry(
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    retryable_exceptions: tuple[Type[Exception], ...] | None = None,
) -> Callable[[F], F]:
    """Decorator for adding retry logic to synchronous functions.
    
    Args:
        config: RetryConfig instance (takes precedence)
        max_attempts: Maximum attempts if not using config
        base_delay: Base delay if not using config
        retryable_exceptions: Exceptions to retry on if not using config
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def call_api():
            return requests.get(url)
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            base_delay=base_delay or 1.0,
            retryable_exceptions=retryable_exceptions,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            "Retry after error",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=config.max_attempts,
                            delay=delay,
                            error=str(e),
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All retry attempts exhausted",
                            function=func.__name__,
                            attempts=config.max_attempts,
                            final_error=str(e),
                        )
            
            # Should not reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")
        
        return wrapper  # type: ignore
    
    return decorator


def async_retry(
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    retryable_exceptions: tuple[Type[Exception], ...] | None = None,
) -> Callable[[F], F]:
    """Decorator for adding retry logic to async functions.
    
    Args:
        config: RetryConfig instance (takes precedence)
        max_attempts: Maximum attempts if not using config
        base_delay: Base delay if not using config
        retryable_exceptions: Exceptions to retry on if not using config
        
    Returns:
        Decorated async function with retry logic
        
    Example:
        @async_retry(max_attempts=3, base_delay=1.0)
        async def call_api():
            return await httpx.get(url)
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            base_delay=base_delay or 1.0,
            retryable_exceptions=retryable_exceptions,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            "Retry after error",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=config.max_attempts,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All retry attempts exhausted",
                            function=func.__name__,
                            attempts=config.max_attempts,
                            final_error=str(e),
                        )
            
            # Should not reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")
        
        return wrapper  # type: ignore
    
    return decorator


# Common retryable exceptions for LLM providers
LLM_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    # These are common HTTP-related errors that might occur
    # Add provider-specific exceptions as needed
)


def with_llm_retry(func: F) -> F:
    """Convenience decorator specifically for LLM API calls.
    
    Uses sensible defaults for LLM APIs:
    - 3 retry attempts
    - Exponential backoff starting at 1 second
    - Retries on timeout and connection errors
    
    Example:
        @with_llm_retry
        async def _call_openai(self, ...):
            ...
    """
    return async_retry(config=DEFAULT_LLM_RETRY_CONFIG)(func)
