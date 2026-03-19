"""Tests for TurnTagIndex integrity during multi-batch ingestion."""

import pytest

from virtual_context.types import Message, TurnTagEntry


class TestCatchUpIngestionNoOverwrite:
    """Verify that catch-up ingestion batches don't overwrite earlier entries."""

    def test_sequential_ingest_no_overwrite(self, tmp_path):
        """Two sequential ingest_history calls must not overwrite each other's entries."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        cfg = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=cfg)

        # Batch 1: 5 turns about cooking
        batch1 = []
        for i in range(5):
            batch1.append(Message(role="user", content=f"Tell me about pasta recipe {i}"))
            batch1.append(Message(role="assistant", content=f"Here's a great pasta recipe {i}"))
        engine.ingest_history(batch1, turn_offset=0)

        assert len(engine._turn_tag_index.entries) == 5
        batch1_tags = [e.tags for e in engine._turn_tag_index.entries]
        batch1_turn0_hash = engine._turn_tag_index.entries[0].message_hash

        # Batch 2: 3 turns about fitness (simulating catch-up gap)
        batch2 = []
        for i in range(3):
            batch2.append(Message(role="user", content=f"What's a good running workout {i}?"))
            batch2.append(Message(role="assistant", content=f"Try interval training {i}"))
        engine.ingest_history(batch2, turn_offset=5)

        # Index should now have 8 entries
        assert len(engine._turn_tag_index.entries) == 8

        # Batch 1 entries must NOT be overwritten
        assert engine._turn_tag_index.entries[0].message_hash == batch1_turn0_hash
        entry0 = engine._turn_tag_index.get_tags_for_turn(0)
        assert entry0 is not None
        assert entry0.message_hash == batch1_turn0_hash

        # Batch 2 entries should have turn_number 5, 6, 7
        assert engine._turn_tag_index.entries[5].turn_number == 5
        assert engine._turn_tag_index.entries[6].turn_number == 6
        assert engine._turn_tag_index.entries[7].turn_number == 7

        # get_tags_for_turn should return correct entries
        entry5 = engine._turn_tag_index.get_tags_for_turn(5)
        assert entry5 is not None
        assert "running" in entry5.message_hash or entry5.turn_number == 5

    def test_turn_offset_zero_no_collision(self, tmp_path):
        """With turn_offset=0, first batch tags should be intact after second batch."""
        from virtual_context.engine import VirtualContextEngine
        from virtual_context.config import load_config

        cfg = load_config(config_dict={
            "context_window": 10000,
            "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "test.db")}},
            "tag_generator": {"type": "keyword"},
        })
        engine = VirtualContextEngine(config=cfg)

        # Initial batch
        batch = [
            Message(role="user", content="What about Emily in Paris?"),
            Message(role="assistant", content="It's a fun show"),
            Message(role="user", content="Bridgerton is better"),
            Message(role="assistant", content="Agreed, period dramas are great"),
        ]
        engine.ingest_history(batch, turn_offset=0)

        turn0_entry = engine._turn_tag_index.get_tags_for_turn(0)
        turn1_entry = engine._turn_tag_index.get_tags_for_turn(1)
        assert turn0_entry is not None
        assert turn1_entry is not None

        # Catch-up batch with different content, correct offset
        gap = [
            Message(role="user", content="Do you read arabic numerals?"),
            Message(role="assistant", content="Yes I can read many scripts"),
        ]
        engine.ingest_history(gap, turn_offset=2)

        # Turn 0 and 1 must still have original tags
        assert engine._turn_tag_index.get_tags_for_turn(0) is turn0_entry
        assert engine._turn_tag_index.get_tags_for_turn(1) is turn1_entry
        # Turn 2 should be the new entry
        turn2_entry = engine._turn_tag_index.get_tags_for_turn(2)
        assert turn2_entry is not None
        assert turn2_entry is not turn0_entry
