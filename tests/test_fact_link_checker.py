"""Tests for FactLinkChecker — expanded from supersession to full link detection."""

from unittest.mock import MagicMock

import pytest

import json

from virtual_context.ingest.supersession import FactLinkChecker
from virtual_context.types import Fact, FactLink, SupersessionConfig
from virtual_context.core.fact_lifecycle import fact_version

_FACTS = {}


@pytest.fixture(autouse=True)
def _reset_fact_inventory():
    _FACTS.clear()


def _make_checker(llm_response="[]", graph_links=True, config=None):
    llm = MagicMock()
    llm.complete.return_value = (llm_response, {})
    llm.last_usage = {}
    store = MagicMock()
    store.query_facts.return_value = []
    store.set_fact_superseded.return_value = True
    store.store_fact_links.side_effect = lambda links, **kwargs: len(links)
    store.get_fact_admission_scope.return_value = None
    store.get_fact_admission_snapshot.side_effect = lambda fact_id: {"fact_version": fact_version(_FACTS[fact_id]), "audience": None, "source_versions": ()}
    cfg = config or SupersessionConfig(enabled=True, batch_size=20)
    checker = FactLinkChecker(
        llm_provider=llm, model="test", store=store, config=cfg,
        graph_links=graph_links,
    )
    return checker, store, llm


def _make_fact(id="f1", subject="user", verb="led", object="Alpha", tags=None, **kw):
    kw.setdefault("conversation_id", "conversation")
    fact = Fact(id=id, subject=subject, verb=verb, object=object, tags=tags or [], **kw)
    _FACTS[id] = fact
    return fact


class TestFactLinkCheckerSupersessionMode:
    """When graph_links=False, behaves identically to old FactSupersessionChecker."""

    def test_supersession_only(self):
        old = _make_fact(id="old", verb="lives-in", object="NYC")
        new = _make_fact(id="new", verb="lives-in", object="Chicago")
        checker, store, llm = _make_checker(llm_response="[0]", graph_links=False)
        store.query_facts.return_value = [old]

        links_created, superseded = checker.check_and_link([new])
        assert superseded == 1
        assert links_created == 0
        # Per fencing plan §5.6 caller-side propagation, FactLinkChecker
        # now forwards guard kwargs (here all None for legacy callers)
        # through to set_fact_superseded.
        store.set_fact_superseded.assert_called_once_with(
            "old", "new",
            operation_id=None, owner_worker_id=None, lifecycle_epoch=None,
            expected_old_version=fact_version(old), expected_new_version=fact_version(new), expected_source_versions=(),
        )

    def test_disabled_config(self):
        cfg = SupersessionConfig(enabled=False)
        checker, store, llm = _make_checker(graph_links=False, config=cfg)
        links, sup = checker.check_and_link([_make_fact()])
        assert links == 0
        assert sup == 0
        llm.complete.assert_not_called()

    def test_empty_facts(self):
        checker, store, llm = _make_checker(graph_links=False)
        links, sup = checker.check_and_link([])
        assert links == 0
        assert sup == 0


class TestFactLinkCheckerGraphMode:
    """When graph_links=True, detects all relationship types and creates links."""

    def test_detects_links(self):
        old = _make_fact(id="old", verb="led", object="Alpha")
        new = _make_fact(id="new", verb="uses", object="Python")
        response = json.dumps({
            "superseded": [],
            "links": [
                {"source": "N0", "target": "E0", "relation": "part_of",
                 "confidence": 0.9, "context": "Python used in Alpha"}
            ]
        })
        checker, store, llm = _make_checker(llm_response=response, graph_links=True)
        store.query_facts.return_value = [old]

        links_created, superseded = checker.check_and_link([new])
        assert superseded == 0
        assert links_created == 1
        store.store_fact_links.assert_called_once()
        stored_links = store.store_fact_links.call_args[0][0]
        assert len(stored_links) == 1
        assert isinstance(stored_links[0], FactLink)
        assert stored_links[0].relation_type == "part_of"
        assert stored_links[0].source_fact_id == "new"
        assert stored_links[0].target_fact_id == "old"

    def test_supersedes_in_graph_mode(self):
        old = _make_fact(id="old", verb="lives-in", object="NYC")
        new = _make_fact(id="new", verb="lives-in", object="Chicago")
        response = json.dumps({
            "superseded": [0],
            "links": [
                {"source": "N0", "target": "E0", "relation": "supersedes",
                 "confidence": 1.0, "context": "Moved from NYC to Chicago"}
            ]
        })
        checker, store, llm = _make_checker(llm_response=response, graph_links=True)
        store.query_facts.return_value = [old]

        links_created, superseded = checker.check_and_link([new])
        assert superseded == 1
        assert links_created == 1
        store.set_fact_superseded.assert_called_once()
        store.store_fact_links.assert_called_once()

    def test_no_links_found(self):
        old = _make_fact(id="old", verb="likes", object="pizza")
        new = _make_fact(id="new", verb="likes", object="sushi")
        response = json.dumps({"superseded": [], "links": []})
        checker, store, llm = _make_checker(llm_response=response, graph_links=True)
        store.query_facts.return_value = [old]

        links_created, superseded = checker.check_and_link([new])
        assert superseded == 0
        assert links_created == 0

    def test_invalid_relation_type_skipped(self):
        old = _make_fact(id="old", verb="led", object="Alpha")
        new = _make_fact(id="new", verb="uses", object="Python")
        response = json.dumps({
            "superseded": [],
            "links": [
                {"source": "N0", "target": "E0", "relation": "INVALID_TYPE",
                 "confidence": 0.5, "context": "bad link"}
            ]
        })
        checker, store, llm = _make_checker(llm_response=response, graph_links=True)
        store.query_facts.return_value = [old]

        links_created, superseded = checker.check_and_link([new])
        assert links_created == 0

    def test_llm_failure_returns_zeros(self):
        old = _make_fact(id="old")
        new = _make_fact(id="new")
        checker, store, llm = _make_checker(graph_links=True)
        store.query_facts.return_value = [old]
        llm.complete.side_effect = RuntimeError("API down")

        links_created, superseded = checker.check_and_link([new])
        assert links_created == 0
        assert superseded == 0
