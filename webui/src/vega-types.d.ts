// Type definitions for Vega-Lite and related modules
declare module "fast-json-patch/index.mjs";

// Add any interface augmentations for TimelineMessage and ToolCall
interface ToolCall {
  name: string;
  args?: string;
  result?: string;
  input?: string; // Add missing property
}

interface TimelineMessage {
  type: string;
  content?: string;
  timestamp?: string | number | Date;
  elapsed?: number;
  end_of_turn?: boolean;
  conversation_id?: string;
  parent_conversation_id?: string;
  tool_calls?: ToolCall[];
  tool_name?: string;
  tool_error?: boolean;
  tool_result?: string;
  input?: string;
  start_time?: string | number | Date; // Add start time
  end_time?: string | number | Date; // Add end time
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
    cost_usd?: number;
  };
}
