# Virtual Context

## WORKFLOW: ONE ISSUE AT A TIME
Work through benchmark issues one at a time. Do not move on from an issue without an explicit directive from the user. Never suggest moving on to the next issue. The user will decide when to move on.

## Environment

Always run tests with `.venv/bin/pytest`. Always use `.venv/bin/python` for any Python commands. The project venv must be used for all operations.

```bash
# Run tests
.venv/bin/pytest tests/

# Run a specific test
.venv/bin/pytest tests/test_foo.py -k "test_name"

# Install dependencies
.venv/bin/pip install -e ".[dev]"
```
