# Virtual Context

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
