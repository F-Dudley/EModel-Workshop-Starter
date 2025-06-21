from multiprocessing import cpu_count


def get_worker_count(leave_amount: int = 0) -> int:
    """
    Returns the number of CPU cores available on the system.

    This function is useful for determining how many worker processes to spawn
    in a multiprocessing context, allowing for efficient parallel processing.

    Returns:
        int: The number of CPU cores available.
    """
    return max(cpu_count() - leave_amount, 1)
