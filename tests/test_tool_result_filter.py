"""Tests for tool_result filtering in extract_history_pairs — assistant responses
after tool_result must not be skipped."""

from virtual_context.proxy.formats import (
    AnthropicFormat,
    summarize_payload_accounting,
    summarize_raw_payload_entries,
)


class TestToolResultFilterPreservesAssistantResponse:
    """When a user message is tool_result-only, the NEXT assistant message
    may contain the real synthesized response. It must not be skipped."""

    def test_assistant_text_after_tool_result_is_preserved(self):
        fmt = AnthropicFormat()
        body = {
            "messages": [
                # msg[0]: real user question
                {"role": "user", "content": "Tell me about rare beauty ingredients"},
                # msg[1]: assistant does web searches (tool_use)
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "I'll search..."},
                    {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"q": "rare beauty"}},
                    {"type": "tool_use", "id": "t2", "name": "web_search", "input": {"q": "revolution"}},
                ]},
                # msg[2]: tool results come back
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Rare Beauty uses..."},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "Revolution uses..."},
                ]},
                # msg[3]: assistant synthesizes results — THIS MUST NOT BE SKIPPED
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Comparing the two..."},
                    {"type": "text", "text": "Same Caprylic/Capric Triglyceride for lightweight moisture"},
                ]},
                # msg[4]: next real user message
                {"role": "user", "content": "Why do you think I have medium skin tone?"},
                # msg[5]: assistant response
                {"role": "assistant", "content": "Based on the shade recommendations..."},
            ]
        }
        pairs = fmt.extract_history_pairs(body)

        # Should have extracted pairs
        assert len(pairs) >= 2

        # "Caprylic" must be in the extracted pairs — it's real assistant content
        all_content = " ".join(p.content for p in pairs)
        assert "Caprylic" in all_content, (
            f"Assistant's Caprylic analysis was dropped by tool_result filter. "
            f"Extracted content: {all_content[:200]}"
        )

    def test_tool_result_content_not_in_pairs(self):
        """The tool_result content itself should not be in the extracted pairs."""
        fmt = AnthropicFormat()
        body = {
            "messages": [
                {"role": "user", "content": "Search for pasta recipes"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "pasta"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": "TOOL_RESULT_MARKER_should_not_appear"},
                ]},
                {"role": "assistant", "content": "Here are some great pasta recipes based on my search"},
                {"role": "user", "content": "Thanks!"},
                {"role": "assistant", "content": "You're welcome!"},
            ]
        }
        pairs = fmt.extract_history_pairs(body)
        all_content = " ".join(p.content for p in pairs)

        assert "TOOL_RESULT_MARKER" not in all_content, (
            "Tool result content leaked into extracted pairs"
        )
        assert "pasta recipes based on my search" in all_content, (
            "Assistant's synthesized response was dropped"
        )

    def test_consecutive_tool_rounds_preserve_final_response(self):
        """Multiple tool_use/tool_result rounds should preserve the final text response."""
        fmt = AnthropicFormat()
        body = {
            "messages": [
                {"role": "user", "content": "Compare iPhone and Pixel cameras"},
                # Round 1: tool_use
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "iphone"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "iPhone specs..."},
                ]},
                # Round 2: more tool_use
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t2", "name": "search", "input": {"q": "pixel"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "Pixel specs..."},
                ]},
                # Final: assistant synthesizes
                {"role": "assistant", "content": "The iPhone 16 Pro has a 48MP main sensor while the Pixel 9 Pro uses 50MP"},
                {"role": "user", "content": "Which is better for low light?"},
                {"role": "assistant", "content": "The Pixel generally wins in low light"},
            ]
        }
        pairs = fmt.extract_history_pairs(body)
        all_content = " ".join(p.content for p in pairs)

        assert "48MP" in all_content, "Final assistant synthesis was dropped"

    def test_payload_accounting_explains_tool_fold_and_pair_counts(self):
        fmt = AnthropicFormat()
        body = {
            "messages": [
                {"role": "user", "content": "Tell me about rare beauty ingredients"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"q": "rare beauty"}},
                    {"type": "tool_use", "id": "t2", "name": "web_search", "input": {"q": "revolution"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Rare Beauty uses..."},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "Revolution uses..."},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Comparing the two..."},
                    {"type": "text", "text": "Same Caprylic/Capric Triglyceride for lightweight moisture"},
                ]},
                {"role": "user", "content": "Why do you think I have medium skin tone?"},
                {"role": "assistant", "content": "Based on the shade recommendations..."},
            ]
        }

        raw = summarize_raw_payload_entries(body["messages"])
        accounting = summarize_payload_accounting(body, fmt, raw_summary=raw)

        assert accounting["raw_payload_entry_count"] == 6
        assert accounting["raw_tool_use_blocks"] == 2
        assert accounting["raw_tool_result_only_user_entries"] == 1
        assert accounting["assistant_tool_only_entries_folded"] == 1
        assert accounting["tool_result_user_entries_folded"] == 1
        assert accounting["tool_followup_pairs"] == 1
        assert accounting["extracted_history_message_count"] == 4
        assert accounting["extracted_history_pair_count"] == 2
