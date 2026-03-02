try:
    from .validation import validate_all
except Exception as e:
    print("VALIDATION IMPORT FAILED:", e)
    raise
