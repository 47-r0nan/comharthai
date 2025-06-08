# Comharthai Tests

This directory contains tests for the Comharthai project.

## Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests for API endpoints and component interactions

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/unit/test_model_factory.py
```

To run tests with verbose output:

```bash
pytest -v
```

## Test Coverage

The current test suite covers:

- Model factory functionality
- API endpoint availability
- Basic API response validation

Note: Some tests are marked with `@pytest.mark.skip` as they require fully implemented models, which are still in development.

## Adding Tests

When adding new functionality, please add corresponding tests following the existing patterns:

1. Unit tests for individual functions and classes
2. Integration tests for API endpoints and workflows
