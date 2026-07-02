## Summary

<!-- Briefly describe what this PR changes. -->

## Related issue

<!-- Link the issue this closes or relates to, for example: Closes #123. -->

## Type of change

- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] Tests
- [ ] Refactor or maintenance

## Testing

<!-- List the exact commands you ran. -->

- [ ] `ruff format --check lettucedetect/ tests/`
- [ ] `ruff check lettucedetect/ tests/ --extend-exclude lettucedetect/integrations/`
- [ ] `pytest tests/test_inference_pytest.py -v -k "not TestAnswerStartToken"`
- [ ] Other:

## Checklist

- [ ] I kept the PR focused on one change.
- [ ] I added or updated tests/docs when needed.
- [ ] I checked that no secrets, API keys, or credentials are included.
