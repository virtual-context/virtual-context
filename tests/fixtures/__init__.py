"""Test fixtures for end-to-end harness scenarios.

Modules in this package produce reproducible Postgres + Redis snapshots
that downstream harnesses (e.g. virtual-context-cloud's VCATTACH end-to-end
harness) restore at test setup to exercise multi-component flows.
"""
