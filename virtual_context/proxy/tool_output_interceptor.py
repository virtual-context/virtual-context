"""Tool output interception — DEPRECATED.

This module previously provided head/tail truncation of large tool outputs.
That approach has been replaced by position-based tool output stubbing
(see proxy/message_filter.py stub_tool_outputs_by_position) which stores
full outputs and replaces them with compact stubs containing restore refs.

The ToolOutputInterceptor class and build_turn_tool_output_refs function
have been removed. All call sites in server.py have been updated.
"""

from __future__ import annotations
