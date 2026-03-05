"""Tests for document ingestion pipeline: parsers, supersession, ingestor, config, CLI."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from virtual_context.config import load_config
from virtual_context.core.store import ContextStore
from virtual_context.ingest.parsers import (
    parse_document,
    parse_text,
    parse_pdf,
    parse_docx,
    parse_xlsx,
)
from virtual_context.ingest.supersession import FactSupersessionChecker
from virtual_context.types import (
    CompactionResult,
    DocumentIngestionConfig,
    DocumentIngestionResult,
    Fact,
    SegmentMetadata,
    SupersessionConfig,
    TagResult,
    VirtualContextConfig,
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class TestParsers:
    def test_parse_text(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("Hello, world!\nLine two.")
        assert parse_text(p) == "Hello, world!\nLine two."

    def test_parse_text_via_dispatch(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("dispatch test")
        assert parse_document(p) == "dispatch test"

    def test_parse_md_via_dispatch(self, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text("# Title\nBody")
        assert parse_document(p) == "# Title\nBody"

    def test_parse_csv_via_dispatch(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        assert parse_document(p) == "a,b\n1,2"

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "image.png"
        p.write_bytes(b"\x89PNG")
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(p)

    def test_parse_pdf_import_error(self, tmp_path):
        p = tmp_path / "test.pdf"
        p.write_bytes(b"%PDF-1.4")
        with patch.dict("sys.modules", {"fitz": None}):
            with pytest.raises(ImportError, match="pymupdf"):
                parse_pdf(p)

    def test_parse_pdf_mock(self, tmp_path):
        p = tmp_path / "test.pdf"
        p.write_bytes(b"%PDF-1.4")

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page1, mock_page2])

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = parse_pdf(p)
        assert "Page 1 content" in result
        assert "Page 2 content" in result

    def test_parse_docx_import_error(self, tmp_path):
        p = tmp_path / "test.docx"
        p.write_bytes(b"PK")
        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                parse_docx(p)

    def test_parse_docx_mock(self, tmp_path):
        p = tmp_path / "test.docx"
        p.write_bytes(b"PK")

        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = ""  # empty paragraph should be skipped
        mock_para3 = MagicMock()
        mock_para3.text = "Third paragraph"

        mock_document_cls = MagicMock()
        mock_document_instance = MagicMock()
        mock_document_instance.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_document_cls.return_value = mock_document_instance

        mock_docx = MagicMock()
        mock_docx.Document = mock_document_cls

        with patch.dict("sys.modules", {"docx": mock_docx}):
            result = parse_docx(p)
        assert "First paragraph" in result
        assert "Third paragraph" in result

    def test_parse_xlsx_import_error(self, tmp_path):
        p = tmp_path / "test.xlsx"
        p.write_bytes(b"PK")
        with patch.dict("sys.modules", {"openpyxl": None}):
            with pytest.raises(ImportError, match="openpyxl"):
                parse_xlsx(p)

    def test_parse_xlsx_mock(self, tmp_path):
        p = tmp_path / "test.xlsx"
        p.write_bytes(b"PK")

        mock_ws = MagicMock()
        mock_ws.iter_rows.return_value = [
            ("Name", "Age"),
            ("Alice", 30),
            (None, None),  # empty row
        ]

        mock_wb = MagicMock()
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.__getitem__ = lambda self, key: mock_ws

        mock_openpyxl = MagicMock()
        mock_openpyxl.load_workbook.return_value = mock_wb

        with patch.dict("sys.modules", {"openpyxl": mock_openpyxl}):
            result = parse_xlsx(p)
        assert "## Sheet1" in result
        assert "Name | Age" in result
        assert "Alice | 30" in result


# ---------------------------------------------------------------------------
# Supersession
# ---------------------------------------------------------------------------

class TestSupersession:
    def _make_fact(self, id: str, subject: str, verb: str, obj: str, status: str = "active") -> Fact:
        return Fact(id=id, subject=subject, verb=verb, object=obj, status=status)

    def test_supersession_disabled(self):
        config = SupersessionConfig(enabled=False)
        llm = MagicMock()
        store = MagicMock()
        checker = FactSupersessionChecker(llm, "model", store, config)
        result = checker.check_and_supersede([self._make_fact("1", "alice", "lives_in", "NYC")])
        assert result == 0
        llm.complete.assert_not_called()

    def test_supersession_no_candidates(self):
        config = SupersessionConfig(enabled=True)
        llm = MagicMock()
        store = MagicMock()
        store.query_facts.return_value = []
        checker = FactSupersessionChecker(llm, "model", store, config)
        result = checker.check_and_supersede([self._make_fact("1", "alice", "lives_in", "NYC")])
        assert result == 0
        llm.complete.assert_not_called()

    def test_supersession_marks_old_fact(self):
        config = SupersessionConfig(enabled=True, batch_size=10)
        old_fact = self._make_fact("old1", "alice", "lives_in", "NYC")
        new_fact = self._make_fact("new1", "alice", "lives_in", "London")

        llm = MagicMock()
        llm.complete.return_value = "[0]"

        store = MagicMock()
        store.query_facts.return_value = [old_fact]

        checker = FactSupersessionChecker(llm, "model", store, config)
        result = checker.check_and_supersede([new_fact])
        assert result == 1
        store.set_fact_superseded.assert_called_once_with("old1", "new1")

    def test_supersession_no_match(self):
        config = SupersessionConfig(enabled=True)
        old_fact = self._make_fact("old1", "alice", "works_at", "Acme")
        new_fact = self._make_fact("new1", "alice", "lives_in", "London")

        llm = MagicMock()
        llm.complete.return_value = "[]"

        store = MagicMock()
        store.query_facts.return_value = [old_fact]

        checker = FactSupersessionChecker(llm, "model", store, config)
        result = checker.check_and_supersede([new_fact])
        assert result == 0
        store.set_fact_superseded.assert_not_called()

    def test_supersession_skips_empty_subject(self):
        config = SupersessionConfig(enabled=True)
        llm = MagicMock()
        store = MagicMock()
        checker = FactSupersessionChecker(llm, "model", store, config)
        fact_no_subject = self._make_fact("1", "", "lives_in", "NYC")
        result = checker.check_and_supersede([fact_no_subject])
        assert result == 0
        store.query_facts.assert_not_called()

    def test_supersession_llm_error_graceful(self):
        config = SupersessionConfig(enabled=True)
        old_fact = self._make_fact("old1", "alice", "lives_in", "NYC")
        new_fact = self._make_fact("new1", "alice", "lives_in", "London")

        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM down")

        store = MagicMock()
        store.query_facts.return_value = [old_fact]

        checker = FactSupersessionChecker(llm, "model", store, config)
        result = checker.check_and_supersede([new_fact])
        assert result == 0

    def test_parse_response_bad_json(self):
        config = SupersessionConfig(enabled=True)
        checker = FactSupersessionChecker(MagicMock(), "m", MagicMock(), config)
        old = self._make_fact("old1", "alice", "lives_in", "NYC")
        assert checker._parse_response("no json here", [old]) == []

    def test_parse_response_out_of_bounds(self):
        config = SupersessionConfig(enabled=True)
        checker = FactSupersessionChecker(MagicMock(), "m", MagicMock(), config)
        old = self._make_fact("old1", "alice", "lives_in", "NYC")
        assert checker._parse_response("[5, 10]", [old]) == []

    def test_supersession_excludes_self(self):
        """A new fact should not be compared against itself."""
        config = SupersessionConfig(enabled=True)
        new_fact = self._make_fact("f1", "alice", "lives_in", "London")

        llm = MagicMock()
        store = MagicMock()
        # query_facts returns the same fact (same id)
        store.query_facts.return_value = [new_fact]

        checker = FactSupersessionChecker(llm, "model", store, config)
        result = checker.check_and_supersede([new_fact])
        assert result == 0
        llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# set_fact_superseded (SQLiteStore)
# ---------------------------------------------------------------------------

class TestSetFactSuperseded:
    def test_sqlite_set_fact_superseded(self, tmp_path):
        from virtual_context.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        fact = Fact(id="fact1", subject="alice", verb="lives_in", object="NYC")
        store.store_facts([fact])

        # Verify it's stored and not superseded
        facts = store.query_facts(subject="alice")
        assert len(facts) == 1
        assert facts[0].superseded_by is None

        # Mark as superseded
        store.set_fact_superseded("fact1", "fact2")

        # Now query_facts (which filters superseded) should return empty
        facts = store.query_facts(subject="alice")
        assert len(facts) == 0


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing:
    def test_default_document_ingestion_config(self):
        config = load_config(config_dict={})
        assert config.document_ingestion.enabled is True
        assert config.document_ingestion.chunk_max_words == 250
        assert config.document_ingestion.chunk_min_words == 20
        assert config.document_ingestion.max_concurrent_chunks == 4
        assert config.document_ingestion.supersession.enabled is True
        assert config.document_ingestion.supersession.provider == ""
        assert config.document_ingestion.supersession.model == ""
        assert config.document_ingestion.supersession.batch_size == 10
        assert config.document_ingestion.supersession.temperature == 0.1

    def test_custom_document_ingestion_config(self):
        config = load_config(config_dict={
            "document_ingestion": {
                "enabled": False,
                "chunk_max_words": 500,
                "chunk_min_words": 50,
                "max_concurrent_chunks": 8,
                "supersession": {
                    "enabled": False,
                    "provider": "anthropic",
                    "model": "claude-haiku",
                    "batch_size": 20,
                    "temperature": 0.5,
                },
            },
        })
        di = config.document_ingestion
        assert di.enabled is False
        assert di.chunk_max_words == 500
        assert di.chunk_min_words == 50
        assert di.max_concurrent_chunks == 8
        assert di.supersession.enabled is False
        assert di.supersession.provider == "anthropic"
        assert di.supersession.model == "claude-haiku"
        assert di.supersession.batch_size == 20
        assert di.supersession.temperature == 0.5


# ---------------------------------------------------------------------------
# Full ingest pipeline (mocked LLM)
# ---------------------------------------------------------------------------

class TestDocumentIngestor:
    def test_full_ingest_pipeline(self, tmp_path):
        """End-to-end ingest with mocked engine components."""
        from virtual_context.ingest.ingestor import DocumentIngestor

        # Create a text file with enough content for multiple chunks
        doc = tmp_path / "test.txt"
        doc.write_text(" ".join(["word"] * 600))  # ~600 words → multiple chunks

        # Mock engine
        engine = MagicMock()
        engine.config = load_config(config_dict={
            "document_ingestion": {
                "supersession": {"enabled": False},
            },
            "summarization": {"provider": "ollama", "model": "test-model"},
            "providers": {"ollama": {"type": "generic_openai"}},
        })

        # Mock store
        mock_store = MagicMock()
        mock_store.get_all_tags.return_value = []
        engine._store = mock_store

        # Mock tag generator
        mock_tag_gen = MagicMock()
        mock_tag_gen.generate_tags.return_value = TagResult(
            tags=["test-tag"], primary="test-tag", source="mock",
        )
        engine._tag_gen = mock_tag_gen

        # Mock compactor — returns one CompactionResult per segment
        def mock_compact(segments, **kwargs):
            results = []
            for seg in segments:
                results.append(CompactionResult(
                    segment_id=seg.id,
                    primary_tag=seg.primary_tag,
                    tags=seg.tags,
                    summary="Summary of chunk",
                    summary_tokens=10,
                    original_tokens=seg.token_count,
                    full_text=seg.messages[0].content if seg.messages else "",
                    messages=[],
                    metadata=SegmentMetadata(),
                    facts=[Fact(subject="test", verb="has", object="value")],
                ))
            return results

        mock_compactor = MagicMock()
        mock_compactor.compact.side_effect = mock_compact
        mock_compactor.model_name = "test-model"
        engine._compactor = mock_compactor

        # Mock token counter
        engine._token_counter = lambda text: len(text.split())

        ingestor = DocumentIngestor(engine)
        result = ingestor.ingest(doc, source_label="test-doc")

        assert result.source_path == str(doc)
        assert result.chunks_created > 0
        assert result.segments_stored == result.chunks_created
        assert result.facts_extracted > 0
        assert result.facts_superseded == 0
        assert result.errors == []
        assert mock_store.store_segment.call_count == result.segments_stored
        assert mock_store.store_facts.call_count == result.segments_stored

    def test_ingest_empty_document(self, tmp_path):
        from virtual_context.ingest.ingestor import DocumentIngestor

        doc = tmp_path / "empty.txt"
        doc.write_text("")

        engine = MagicMock()
        engine.config = load_config(config_dict={})

        ingestor = DocumentIngestor(engine)
        result = ingestor.ingest(doc)

        assert result.chunks_created == 0
        assert "empty" in result.errors[0].lower()

    def test_ingest_unsupported_format(self, tmp_path):
        from virtual_context.ingest.ingestor import DocumentIngestor

        doc = tmp_path / "image.bmp"
        doc.write_bytes(b"BM")

        engine = MagicMock()
        engine.config = load_config(config_dict={})

        ingestor = DocumentIngestor(engine)
        result = ingestor.ingest(doc)

        assert result.chunks_created == 0
        assert len(result.errors) == 1
        assert "Unsupported" in result.errors[0]


# ---------------------------------------------------------------------------
# CLI cmd_ingest
# ---------------------------------------------------------------------------

class TestCLIIngest:
    def test_cmd_ingest_file_not_found(self, capsys):
        from virtual_context.cli.main import cmd_ingest

        args = MagicMock()
        args.path = "/nonexistent/file.txt"
        args.config = None
        args.label = None

        with pytest.raises(SystemExit, match="1"):
            cmd_ingest(args)

        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_cmd_ingest_success(self, tmp_path, capsys):
        from virtual_context.cli.main import cmd_ingest

        doc = tmp_path / "test.txt"
        doc.write_text("Some test content for ingestion")

        args = MagicMock()
        args.path = str(doc)
        args.config = None
        args.label = "test-label"

        mock_result = DocumentIngestionResult(
            source_path=str(doc),
            chunks_created=3,
            segments_stored=3,
            facts_extracted=5,
            facts_superseded=1,
        )

        mock_engine = MagicMock()
        mock_engine.ingest_document.return_value = mock_result

        with patch("virtual_context.engine.VirtualContextEngine", return_value=mock_engine):
            cmd_ingest(args)

        captured = capsys.readouterr()
        assert "Ingested:" in captured.out
        assert "Chunks:     3" in captured.out
        assert "Segments:   3" in captured.out
        assert "Facts:      5" in captured.out
        assert "Superseded: 1" in captured.out


# ---------------------------------------------------------------------------
# DocumentIngestionResult dataclass
# ---------------------------------------------------------------------------

class TestDocumentIngestionResult:
    def test_defaults(self):
        result = DocumentIngestionResult()
        assert result.source_path == ""
        assert result.chunks_created == 0
        assert result.errors == []

    def test_with_errors(self):
        result = DocumentIngestionResult(errors=["parse failed"])
        assert len(result.errors) == 1
