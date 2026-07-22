from virtual_context.core.canonical_resequence import plan_canonical_resequence


def _row(
    row_id, origin, sort_key, timestamp, user="", assistant="", turn_group=0,
):
    return {
        "canonical_turn_id": row_id,
        "origin_conversation_id": origin,
        "sort_key": sort_key,
        "turn_group_number": turn_group,
        "first_seen_at": timestamp,
        "last_seen_at": timestamp,
        "created_at": timestamp,
        "updated_at": timestamp,
        "user_content": user,
        "assistant_content": assistant,
    }


def test_resequence_keeps_source_pairs_and_interleaves_by_timestamp():
    rows = [
        _row("target-u", "", 1000, "2026-07-20T12:00:00+00:00", user="late"),
        _row("target-a", "", 2000, "2026-07-20T12:00:01+00:00", assistant="reply"),
        _row("source-u", "source", 3000, "2026-07-19T12:00:00+00:00", user="early"),
        _row("source-a", "source", 4000, "2026-07-19T12:00:01+00:00", assistant="reply"),
    ]

    assignments, mapping = plan_canonical_resequence(
        rows, owner_conversation_id="target",
    )

    by_id = {item.canonical_turn_id: item for item in assignments}
    assert by_id["source-u"].turn_group_number == 0
    assert by_id["source-a"].turn_group_number == 0
    assert by_id["target-u"].turn_group_number == 1
    assert by_id["target-a"].turn_group_number == 1
    assert mapping == {("source", 0): 0, ("target", 0): 1}


def test_resequence_does_not_merge_reused_source_group_numbers():
    rows = [
        _row("u1", "source", 1000, "2026-07-19T12:00:00+00:00", user="one"),
        _row("a1", "source", 2000, "2026-07-19T12:00:01+00:00", assistant="one"),
        _row("u2", "source", 3000, "2026-07-20T12:00:00+00:00", user="two"),
        _row("a2", "source", 4000, "2026-07-20T12:00:01+00:00", assistant="two"),
    ]

    assignments, mapping = plan_canonical_resequence(
        rows, owner_conversation_id="target",
    )

    by_id = {item.canonical_turn_id: item for item in assignments}
    assert by_id["u1"].turn_group_number == by_id["a1"].turn_group_number
    assert by_id["u2"].turn_group_number == by_id["a2"].turn_group_number
    assert by_id["u1"].turn_group_number != by_id["u2"].turn_group_number
    assert ("source", 0) not in mapping


def test_resequence_does_not_alias_missing_group_to_turn_zero():
    rows = [
        _row("u", "source", 1000, "2026-07-19T12:00:00+00:00", user="one"),
        _row("a", "source", 2000, "2026-07-19T12:00:01+00:00", assistant="one"),
        _row(
            "unknown", "source", 3000, "2026-07-19T12:01:00+00:00",
            turn_group=None,
        ),
    ]

    assignments, mapping = plan_canonical_resequence(
        rows, owner_conversation_id="target",
    )

    assert mapping[("source", 0)] == 0
    assert all(key[1] >= 0 for key in mapping)
    assert {
        item.canonical_turn_id: item.original_turn_group_number
        for item in assignments
    }["unknown"] == -1


def test_resequence_keeps_provenance_only_row_inside_pending_pair():
    rows = [
        _row("u", "source", 1000, "2026-07-19T12:00:00+00:00", user="one"),
        _row("metadata", "source", 2000, "", turn_group=0),
        _row("a", "source", 3000, "2026-07-19T12:00:01+00:00", assistant="one"),
    ]

    assignments, mapping = plan_canonical_resequence(
        rows, owner_conversation_id="target",
    )

    assert {item.turn_group_number for item in assignments} == {0}
    assert mapping == {("source", 0): 0}


def test_resequence_does_not_pair_adjacent_roles_from_distinct_old_groups():
    rows = [
        _row(
            "u", "source", 1000, "2026-07-19T12:00:00+00:00",
            user="one", turn_group=4,
        ),
        _row(
            "a", "source", 2000, "2026-07-19T12:00:01+00:00",
            assistant="separate", turn_group=5,
        ),
    ]

    assignments, mapping = plan_canonical_resequence(
        rows, owner_conversation_id="target",
    )

    assert [item.turn_group_number for item in assignments] == [0, 1]
    assert mapping == {("source", 4): 0, ("source", 5): 1}


def test_resequence_undated_group_inherits_neighbor_instead_of_becoming_newest():
    rows = [
        _row("dated", "a", 1000, "2026-07-19T12:00:00+00:00", user="dated"),
        _row("undated", "a", 2000, "", user="undated", turn_group=1),
        _row("later", "b", 1000, "2026-07-20T12:00:00+00:00", user="later"),
    ]

    assignments, _mapping = plan_canonical_resequence(
        rows, owner_conversation_id="target",
    )

    by_id = {item.canonical_turn_id: item for item in assignments}
    assert by_id["dated"].turn_group_number == 0
    assert by_id["undated"].turn_group_number == 1
    assert by_id["later"].turn_group_number == 2
