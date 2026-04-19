#!/usr/bin/env bash
# Section-level pytest timing report.
# Usage: ./tests_by_section.sh [section_name_or_all]
# Outputs: per-section wall time + top 10 slowest tests.
# Output file: /tmp/pytest_section_report.txt

set -u

REPORT=/tmp/pytest_section_report.txt
: > "$REPORT"
PY=.venv/bin/python

run_section() {
    local name="$1"
    shift
    local files=("$@")
    local existing=()
    for f in "${files[@]}"; do
        if [ -f "$f" ]; then existing+=("$f"); fi
    done
    if [ ${#existing[@]} -eq 0 ]; then
        echo "[SECTION=$name] NO FILES"  | tee -a "$REPORT"
        return
    fi
    echo "======================================================" | tee -a "$REPORT"
    echo "[SECTION=$name] ${#existing[@]} files" | tee -a "$REPORT"
    echo "======================================================" | tee -a "$REPORT"
    local t0=$(date +%s)
    # Use --timeout to prevent hangs; --durations=15 to see slow tests
    $PY -m pytest "${existing[@]}" -q --durations=15 --timeout=120 2>&1 | tail -80 | tee -a "$REPORT"
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    echo "[SECTION=$name] WALL_TIME=${elapsed}s" | tee -a "$REPORT"
    echo "" | tee -a "$REPORT"
}

# Section definitions. File lists reflect current tests/ layout.
STORAGE_SCHEMA=(
    tests/test_store_sqlite.py
    tests/test_postgres_store.py
    tests/test_canonical_turns_schema.py
    tests/test_canonical_turns_schema_postgres.py
    tests/test_conversation_schema.py
    tests/test_conversation_schema_postgres.py
    tests/test_session_state.py
    tests/test_lifecycle_epoch_store.py
    tests/test_lifecycle_epoch_store_postgres.py
    tests/test_ingestion_episode_crud.py
    tests/test_ingestion_episode_crud_postgres.py
    tests/test_compaction_operation_crud.py
    tests/test_compaction_operation_crud_postgres.py
    tests/test_canonical_row_epoch_guards.py
    tests/test_canonical_row_epoch_guards_postgres.py
    tests/test_progress_snapshot.py
    tests/test_progress_snapshot_postgres.py
    tests/test_composite_store.py
    tests/test_storage_protocols.py
    tests/test_request_metadata.py
    tests/test_request_metadata_postgres.py
    tests/test_noop_fact_link_store.py
    tests/test_fact_links_sqlite.py
    tests/test_store_recovery.py
)

LIFECYCLE_EPOCH=(
    tests/test_lifecycle_epoch.py
    tests/test_engine_lifecycle_epoch.py
    tests/test_delete_resurrect_epoch_races.py
    tests/test_canonical_row_epoch_guards.py
    tests/test_canonical_row_epoch_guards_postgres.py
)

INGEST=(
    tests/test_engine_sync_turns.py
    tests/test_ingest_reconciler_epoch.py
    tests/test_ingest_index_integrity.py
    tests/test_multi_worker_sliding_window.py
    tests/test_handle_prepare_payload.py
    tests/test_ingestion_episode.py
    tests/test_ingestion_episode_postgres.py
    tests/test_ingestion_progress_event.py
)

TAGGING=(
    tests/test_tag_generator.py
    tests/test_tag_splitter.py
    tests/test_tag_canonicalizer.py
    tests/test_tag_consolidator.py
    tests/test_tagger_loop.py
    tests/test_tool_tags.py
    tests/test_turn_tag_index.py
    tests/test_turn_grouping.py
    tests/test_embedding_tag_generator.py
    tests/test_canonical_turn_grouping.py
)

COMPACTION=(
    tests/test_compaction_commit_prune.py
    tests/test_compaction_lifecycle.py
    tests/test_compaction_lifecycle_wiring.py
    tests/test_compaction_operation.py
    tests/test_compaction_operation_postgres.py
    tests/test_compaction_progress_event.py
    tests/test_compactor_concurrent.py
    tests/test_compactor.py
    tests/test_deferred_payload_compaction.py
)

PROXY=(
    tests/test_proxy.py
    tests/test_proxy_dashboard.py
    tests/test_proxy_formats.py
    tests/test_proxy_message_filter.py
    tests/test_proxy_session.py
    tests/test_proxy_streaming.py
)

ENGINE=(
    tests/test_engine_event_bus.py
    tests/test_engine_integration.py
    tests/test_engine_lookback.py
    tests/test_engine_state.py
)

RETRIEVAL=(
    tests/test_assembler.py
    tests/test_retriever.py
    tests/test_semantic_search.py
    tests/test_segmenter.py
    tests/test_idf_retrieval.py
    tests/test_rrf_scoring.py
    tests/test_recall_all.py
    tests/test_fill_pass.py
    tests/test_find_quote.py
    tests/test_temporal_resolver.py
    tests/test_verb_expansion.py
    tests/test_paging.py
)

PROGRESS_EVENTS=(
    tests/test_progress_event_bus.py
    tests/test_progress_events.py
    tests/test_phase_transition_events.py
)

FILTERS_FORMATS=(
    tests/test_format_agnostic.py
    tests/test_message_filter.py
    tests/test_history_filter.py
    tests/test_passthrough_filter.py
    tests/test_tool_result_filter.py
    tests/test_media.py
    tests/test_raw_content.py
    tests/test_upstream_trim.py
    tests/test_stub_turn_handling.py
    tests/test_empty_turn_skip.py
    tests/test_context_bleed.py
    tests/test_prev_context_leak.py
)

FACTS=(
    tests/test_fact_enrichment.py
    tests/test_fact_graph_integration.py
    tests/test_fact_link_checker.py
    tests/test_fact_link_query.py
    tests/test_fact_link_types.py
    tests/test_fact_redesign.py
)

CONFIG_MODEL=(
    tests/test_config.py
    tests/test_model_catalog.py
    tests/test_model_limits.py
    tests/test_presets.py
    tests/test_cli_init.py
    tests/test_budget_enforcement.py
    tests/test_unified_budget.py
)

MISC=(
    tests/test_vcattach.py
    tests/test_monitor.py
    tests/test_headless.py
    tests/test_telemetry.py
    tests/test_telemetry_integration.py
    tests/test_registry_lifecycle.py
    tests/test_multi_instance.py
    tests/test_metrics_persistence.py
    tests/test_longmemeval_auth.py
    tests/test_beam_judge.py
    tests/test_mcp_server.py
    tests/test_request_captures_persistence.py
    tests/test_backend_integration.py
    tests/test_provider_adapters.py
    tests/test_openrouter_provider.py
    tests/test_conversation_identity.py
    tests/test_conversation_lifecycle.py
    tests/test_conversation_scoping.py
    tests/test_conversation_coverage_report.py
    tests/test_session_cache.py
    tests/test_session_date.py
    tests/test_sender_identity.py
    tests/test_tool_loop.py
    tests/test_tool_output_interceptor.py
    tests/test_supersession.py
    tests/test_supersession_migration.py
)

target="${1:-all}"

case "$target" in
    storage) run_section storage-schema "${STORAGE_SCHEMA[@]}" ;;
    lifecycle) run_section lifecycle-epoch "${LIFECYCLE_EPOCH[@]}" ;;
    ingest) run_section ingest "${INGEST[@]}" ;;
    tagging) run_section tagging "${TAGGING[@]}" ;;
    compaction) run_section compaction "${COMPACTION[@]}" ;;
    proxy) run_section proxy "${PROXY[@]}" ;;
    engine) run_section engine "${ENGINE[@]}" ;;
    retrieval) run_section retrieval "${RETRIEVAL[@]}" ;;
    progress) run_section progress-events "${PROGRESS_EVENTS[@]}" ;;
    filters) run_section filters-formats "${FILTERS_FORMATS[@]}" ;;
    facts) run_section facts "${FACTS[@]}" ;;
    config) run_section config-model "${CONFIG_MODEL[@]}" ;;
    misc) run_section misc "${MISC[@]}" ;;
    all)
        run_section storage-schema "${STORAGE_SCHEMA[@]}"
        run_section lifecycle-epoch "${LIFECYCLE_EPOCH[@]}"
        run_section ingest "${INGEST[@]}"
        run_section tagging "${TAGGING[@]}"
        run_section compaction "${COMPACTION[@]}"
        run_section proxy "${PROXY[@]}"
        run_section engine "${ENGINE[@]}"
        run_section retrieval "${RETRIEVAL[@]}"
        run_section progress-events "${PROGRESS_EVENTS[@]}"
        run_section filters-formats "${FILTERS_FORMATS[@]}"
        run_section facts "${FACTS[@]}"
        run_section config-model "${CONFIG_MODEL[@]}"
        run_section misc "${MISC[@]}"
        ;;
    *)
        echo "Unknown section: $target"
        echo "Valid: storage lifecycle ingest tagging compaction proxy engine retrieval progress filters facts config misc all"
        exit 1
        ;;
esac

echo "==============================================" | tee -a "$REPORT"
echo "Report written to: $REPORT"                    | tee -a "$REPORT"
