import time
import uuid


def generate_timestamp():
    now = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{now}_{short_uuid}"