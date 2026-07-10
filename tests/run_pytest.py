#!/usr/bin/env python
"""Script to run pytest tests for the lettucedetect package."""

import sys

import pytest


def run_tests():
    """Run pytest tests for the lettucedetect package."""
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(run_tests())
