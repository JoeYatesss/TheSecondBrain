from openai import RateLimitError
import random
import time


class MaxRetriesExceeded(Exception):
    """Raised when maximum number of retries is exceeded."""
    pass


class Retry:
    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def call_with_retry(self, func):
        for attempt in range(self.max_retries):
            try:
                return func()
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    # Last attempt failed, re-raise the error
                    raise
                # Exponential backoff: 2^attempt seconds + random jitter
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
        
        # This should never be reached due to the raise above, but kept for safety
        raise MaxRetriesExceeded(f"Failed after {self.max_retries} attempts")

