import time
import uuid
import argparse


def generate_timestamp():
    now = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{now}_{short_uuid}"

def positive_int(value: str) -> int:
    try:
        number = int(value)
    except value:
        raise argparse.ArgumentTypeError(f"Expected an integer") from None

    if number <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive")

    return number

def positive_float(value: str) -> float:
    try:
        number = float(value)
    except value:
        raise argparse.ArgumentTypeError(f"Expected an float") from None

    if number <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive")

    return number