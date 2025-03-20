try:
    from agents.tracing.processors import BatchTraceProcessor, BackendSpanExporter
    from agents.tracing.traces import Trace
    from agents.tracing.spans import SpanImpl, Span
    from agents.tracing.span_data import (
        ResponseSpanData,
        FunctionSpanData,
        GenerationSpanData,
        HandoffSpanData,
        CustomSpanData,
        AgentSpanData,
        GuardrailSpanData,
    )
    import httpx
    import copy

    AGENTS_AVAILABLE = True
except ImportError:
    raise ImportError(
        "OpenAI agents integration requires additional dependencies. "
        "Please install them with: pip install 'Arena-tracing[openai-agents]'"
    )


from curses import halfdelay
import datetime
from tkinter import Message
from typing import Any, Dict, Optional, Union, List
from openai.types.responses.response_output_item import (
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
    ResponseFunctionWebSearch
)
from openai.types.responses.response_input_item_param import (
    ResponseFunctionToolCallParam,
    FunctionCallOutput
)
import random
import time
import logging
import os
from collections import defaultdict
from colorama import Fore, Style, init
import json

# Initialize colorama
init()

# Configure the arena logger
try:
    from backend.logger import openai_agent_logger as logger
except ImportError:
    logger = logging.getLogger("arena_logger")
    logger.setLevel(logging.INFO)

# Optionally add a stream handler if you still want console output
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(stream_handler)

# Timeline visualization class
class TraceVisualizer:
    """
    A class to visualize traces as a timeline in the terminal.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceVisualizer, cls).__new__(cls)
            cls._instance._spans = []
            cls._instance._traces = defaultdict(list)
            cls._instance._colors = {
                "response": Fore.GREEN,
                "function": Fore.BLUE,
                "generation": Fore.YELLOW,
                "handoff": Fore.MAGENTA,
                "custom": Fore.CYAN,
                "agent": Fore.RED,
                "guardrail": Fore.LIGHTRED_EX,
            }
            cls._instance._default_color = Fore.WHITE
        return cls._instance
    
    def add_span(self, span_data: Dict[str, Any]):
        """Add a span to the visualizer."""
        if not span_data:
            return
        
        self._spans.append(span_data)
        trace_id = span_data.get('trace_unique_id')
        if trace_id:
            self._traces[trace_id].append(span_data)
    
    def _get_span_color(self, span_name: str):
        """Get color for a span based on its type."""
        for span_type, color in self._colors.items():
            if span_type in span_name.lower():
                return color
        return self._default_color
    
    def _format_time(self, time_obj):
        """Format datetime object to a readable string."""
        if isinstance(time_obj, datetime.datetime):
            return time_obj.strftime("%H:%M:%S.%f")[:-3]
        elif isinstance(time_obj, str):
            # Try to parse string as datetime
            try:
                # Try ISO format first
                dt = datetime.datetime.fromisoformat(time_obj.replace('Z', '+00:00'))
                return dt.strftime("%H:%M:%S.%f")[:-3]
            except (ValueError, TypeError):
                # Try other common formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                    try:
                        dt = datetime.datetime.strptime(time_obj, fmt)
                        return dt.strftime("%H:%M:%S.%f")[:-3]
                    except ValueError:
                        continue
                # If we can't parse, return the string as is
                return time_obj
        elif isinstance(time_obj, (int, float)):
            # Assume Unix timestamp
            try:
                dt = datetime.datetime.fromtimestamp(time_obj)
                return dt.strftime("%H:%M:%S.%f")[:-3]
            except:
                return str(time_obj)
        return str(time_obj)
    
    def _calculate_duration(self, start, end):
        """Calculate duration between start and end times."""
        # Handle datetime objects
        if isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime):
            return (end - start).total_seconds() * 1000  # in milliseconds
        
        # Handle string timestamps (try to convert to datetime)
        if isinstance(start, str) and isinstance(end, str):
            try:
                # Try ISO format first
                start_dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.datetime.fromisoformat(end.replace('Z', '+00:00'))
                return (end_dt - start_dt).total_seconds() * 1000
            except (ValueError, TypeError):
                try:
                    # Try other common formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                        try:
                            start_dt = datetime.datetime.strptime(start, fmt)
                            end_dt = datetime.datetime.strptime(end, fmt)
                            return (end_dt - start_dt).total_seconds() * 1000
                        except ValueError:
                            continue
                except:
                    pass
        
        # Handle numeric timestamps (Unix timestamps)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return (end - start) * 1000  # Assuming seconds, convert to ms
            
        # If we can't calculate, return a small positive value
        return 0.1  # Return a small positive value instead of 0
    
    def visualize_latest_trace(self):
        """Visualize the most recent trace as a timeline in the terminal."""
        if not self._spans:
            print("No spans to visualize.")
            return
        
        # Get the most recent trace
        latest_trace_id = self._spans[-1].get('trace_unique_id')
        if not latest_trace_id or latest_trace_id not in self._traces:
            print("No valid trace to visualize.")
            return
        
        trace_spans = self._traces[latest_trace_id]
        if not trace_spans:
            print("No spans in the trace to visualize.")
            return
            
        # Sort spans by start time
        trace_spans.sort(key=lambda x: x.get('start_time', 0))
        
        # Find the earliest and latest times
        earliest_time = trace_spans[0].get('start_time')
        latest_time = max(span.get('timestamp', span.get('start_time')) for span in trace_spans)
        
        # Calculate total duration
        total_duration = self._calculate_duration(earliest_time, latest_time)
        
        # Handle case where duration is zero or invalid
        if total_duration <= 0:
            # If we only have one span or spans with same start/end time, 
            # assign a minimal duration to still show something
            print("Trace has zero or invalid duration. Using default visualization width.")
            total_duration = 1.0  # Use a default duration of 1ms
            
            # For single span case, we'll just show a fixed width bar
            single_span_mode = True
        else:
            single_span_mode = False
        
        # Terminal width for scaling
        term_width = os.get_terminal_size().columns - 40  # Leave space for text
        
        print("\n" + "=" * 80)
        print(f"TRACE TIMELINE: {latest_trace_id}")
        print("=" * 80)
        
        # Display each span as a line in the timeline
        for i, span in enumerate(trace_spans):
            span_id = span.get('span_unique_id', 'unknown')
            span_name = span.get('span_name', 'unknown')
            start_time = span.get('start_time')
            end_time = span.get('timestamp', start_time)  # Default to start_time if timestamp is missing
            
            # Calculate position and width
            if single_span_mode:
                # In single span mode, distribute spans evenly
                start_pos = int((i / max(1, len(trace_spans))) * term_width * 0.8)
                width = max(5, int(term_width * 0.2))  # Use 20% of terminal width
            else:
                # Normal mode - calculate based on time
                start_offset = self._calculate_duration(earliest_time, start_time)
                duration = self._calculate_duration(start_time, end_time)
                
                # Ensure we have a positive duration
                duration = max(0.1, duration)  # Minimum 0.1ms to be visible
                
                start_pos = int((start_offset / total_duration) * term_width)
                width = max(1, int((duration / total_duration) * term_width))
            
            # Create the timeline bar
            timeline = " " * start_pos + self._get_span_color(span_name) + "█" * width + Style.RESET_ALL
            
            # Format the output line
            start_time_str = self._format_time(start_time)
            
            # Calculate and format duration
            if isinstance(start_time, datetime.datetime) and isinstance(end_time, datetime.datetime):
                duration_ms = (end_time - start_time).total_seconds() * 1000
                duration_str = f"{duration_ms:.2f}ms"
            else:
                duration_str = "unknown"
            
            print(f"{start_time_str} | {span_name[:15]:<15} | {duration_str:>8} | {timeline}")
        
        print("=" * 80 + "\n")

# Add this helper function after the logger configuration
def _truncate_base64_images(data, max_length=100):
    """
    Truncate base64 image data in logs to prevent excessively long log entries.
    
    Args:
        data: The data to process (can be dict, list, string, or other types)
        max_length: Maximum length to keep for base64 image strings
        
    Returns:
        Processed data with truncated base64 strings
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = _truncate_base64_images(value, max_length)
        return result
    elif isinstance(data, list):
        return [_truncate_base64_images(item, max_length) for item in data]
    elif isinstance(data, str):
        # Check if this is likely a base64 image
        if len(data) > 500 and (
            data.startswith("data:image") or 
            (len(data) > 1000 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in data[:100]))
        ):
            return f"{data[:max_length]}... [TRUNCATED BASE64 IMAGE - {len(data)} chars]"
        return data
    else:
        return data

# Define ENV_TYPE for environment detection
ENV_TYPE = ""
try:
    from backend.agents.agent_manager import AgentManager
    ENV_TYPE = "deploy"
except ImportError:
    try:
        from agent_manager import AgentManager
        ENV_TYPE = "local"
    except ImportError:
        ENV_TYPE = "unknown"
        pass

class SpanData():
    trace_unique_id: str
    span_unique_id: str
    span_parent_id: str
    start_time: datetime
    timestamp: datetime
    error_bit: int
    status_code: int
    latency: float
    span_name: str
    span_tools: List[str]
    span_handoffs: List[str]
    metadata: Dict[str, Any] = {}
    
    # Response data attributes
    prompt_messages: List[Any] = None
    tool_calls: List[Any] = None
    tools: List[Any] = None
    input: Any = None
    output: Any = None
    prompt_tokens: int = None
    completion_tokens: int = None
    total_request_tokens: int = None
    model: str = None
    completion_messages: List[Any] = None
    completion_message: Any = None
    full_response: Dict[str, Any] = None
    
    # Function data attributes
    log_type: str = None
    
    # Agent data attributes
    span_workflow_name: str = None
    
    # Generation data attributes
    temperature: float = None
    max_tokens: int = None
    top_p: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    
    # Guardrail data attributes
    has_warnings: bool = None
    warnings_dict: Dict[str, str] = None
    
    span_tools: Any = None
    span_handoffs: Any = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, mode: str = "json"):
        if mode == "json":
            return self.__dict__
        else:
            return self.__dict__

class ArenaSpanExporter(BackendSpanExporter):
    """
    Custom exporter for Keywords AI that handles all span types and allows for dynamic endpoint configuration.
    
    IMPORTANT IMPLEMENTATION NOTE:
    This exporter and all its methods must NEVER modify the original span data objects.
    Always create copies of data structures before manipulation to avoid side effects
    in the agent context. All the methods should treat the span_data parameters as read-only.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        endpoint: str = "https://api.Arena.co/api/",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        agent = None,
        agent_manager = None,
    ):
        """
        Initialize the Keywords AI exporter.

        Args:
            api_key: The API key for authentication. Defaults to os.environ["OPENAI_API_KEY"] if not provided.
            organization: The organization ID. Defaults to os.environ["OPENAI_ORG_ID"] if not provided.
            project: The project ID. Defaults to os.environ["OPENAI_PROJECT_ID"] if not provided.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
            agent: The agent instance associated with this exporter.
            agent_manager: The agent manager instance for sending messages to frontend.
        """
        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            endpoint=endpoint,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        self.agent = agent
        self.agent_manager = agent_manager

    def _format_annotations(self, annotations):
        """
        Format annotations in a readable way.
        
        Args:
            annotations: List of annotation objects
            
        Returns:
            Dictionary with formatted annotation data
        """
        formatted_annotations = []
        
        for annotation in annotations:
            annotation_data = {}
            
            # Common properties
            if hasattr(annotation, "type"):
                annotation_data["type"] = annotation.type
            if hasattr(annotation, "start_index"):
                annotation_data["start_index"] = annotation.start_index
            if hasattr(annotation, "end_index"):
                annotation_data["end_index"] = annotation.end_index
                
            # URL citation specific properties
            if hasattr(annotation, "url"):
                annotation_data["url"] = annotation.url
            if hasattr(annotation, "title"):
                annotation_data["title"] = annotation.title
                
            # File citation specific properties
            if hasattr(annotation, "file_citation"):
                annotation_data["file_citation"] = annotation.file_citation
                
            # File path specific properties
            if hasattr(annotation, "file_path"):
                annotation_data["file_path"] = annotation.file_path
                
            formatted_annotations.append(annotation_data)
            
        return formatted_annotations
        
    def _extract_text_with_citations(self, text, annotations):
        """
        Extract text with citation markers.
        
        Args:
            text: The original text
            annotations: List of annotation objects
            
        Returns:
            Text with citation markers
        """
        if not text or not annotations:
            return text
            
        # Sort annotations by start_index in reverse order to avoid index shifting
        sorted_annotations = sorted(
            annotations, 
            key=lambda a: getattr(a, "start_index", 0), 
            reverse=True
        )
        
        # Make a copy of the text
        text_with_citations = text
        
        # Insert citation markers
        for annotation in sorted_annotations:
            start_index = getattr(annotation, "start_index", None)
            end_index = getattr(annotation, "end_index", None)
            
            if start_index is None or end_index is None:
                continue
                
            citation_type = getattr(annotation, "type", "unknown")
            
            # Create citation marker based on type
            if citation_type == "url_citation":
                url = getattr(annotation, "url", "")
                title = getattr(annotation, "title", "")
                citation_marker = f"[{title}]({url})"
            elif citation_type == "file_citation":
                file_citation = getattr(annotation, "file_citation", {})
                citation_marker = f"[File: {file_citation.get('title', 'Unknown')}]"
            elif citation_type == "file_path":
                file_path = getattr(annotation, "file_path", {})
                citation_marker = f"[File Path: {file_path.get('title', 'Unknown')}]"
            else:
                citation_marker = f"[Citation: {citation_type}]"
                
            # Insert citation marker after the cited text
            text_with_citations = (
                text_with_citations[:end_index] + 
                f" {citation_marker}" + 
                text_with_citations[end_index:]
            )
            
        return text_with_citations
        
    def _extract_web_search_data(self, response_items):
        """
        Extract web search data from response items.
        
        IMPORTANT: This function should only READ from response_items and NEVER MODIFY them
        to avoid side effects in the agent context.
        
        Args:
            response_items: List of response items from the OpenAI API (treat as read-only)
            
        Returns:
            Dictionary containing extracted text and annotations
        """
        # Create a new dictionary to store the extracted data
        # Never modify the original response_items
        result = {
            "text": None,
            "annotations": [],
            "web_search_found": False
        }
        
        logger.info("Extracting web search data from response items")
        
        for item in response_items:
            logger.info(f"Processing response item of type: {type(item).__name__}")
            
            if isinstance(item, ResponseFunctionWebSearch):
                logger.info(f"Found ResponseFunctionWebSearch: id={item.id}, status={item.status}")
                result["web_search_found"] = True
                result["web_search_id"] = item.id
                result["web_search_status"] = item.status
                result["web_search_type"] = item.type
            
            elif isinstance(item, ResponseOutputMessage) and item.content:
                logger.info(f"Found ResponseOutputMessage with {len(item.content)} content items")
                for content_item in item.content:
                    logger.info(f"Processing content item of type: {type(content_item).__name__}")
                    
                    if hasattr(content_item, "text"):
                        result["text"] = content_item.text
                        logger.info(f"Extracted text (first 100 chars): {content_item.text[:100]}...")
                    
                    if hasattr(content_item, "annotations") and content_item.annotations:
                        # Store a copy of the annotations list, not the original reference
                        result["annotations"] = list(content_item.annotations)
                        logger.info(f"Found {len(content_item.annotations)} annotations")
                        
                        # Format annotations for better readability
                        formatted_annotations = self._format_annotations(content_item.annotations)
                        result["formatted_annotations"] = formatted_annotations
                        
                        # # Extract text with citation markers
                        # result["text_with_citations"] = self._extract_text_with_citations(
                        #     result["text"], 
                        #     content_item.annotations
                        # )
                        
                        for i, annotation in enumerate(formatted_annotations):
                            logger.info(f"Annotation {i+1}: {annotation}")
        
        logger.info(f"Web search data extraction complete. Found: {result['web_search_found']}")
        return result

    def _extract_response_message_data(self, response_items):
        """
        Extract response message data from response items.
        
        IMPORTANT: This function should only READ from response_items and NEVER MODIFY them
        to avoid side effects in the agent context.
        
        Args:
            response_items: List of response items from the OpenAI API (treat as read-only)
            
        Returns:
            Dictionary containing a flag indicating if only ResponseOutputMessage was found
            and the extracted text if applicable
        """
        # Create a new dictionary to store the extracted data
        result = {
            "response_message_found": False,
            "text": None
        }
        
        logger.info("Extracting response message data from response items")
        
        # Check if all items are ResponseOutputMessage
        if not response_items:
            logger.info("No response items found")
            return result
            
        # Check if all items are ResponseOutputMessage type
        all_output_messages = all(isinstance(item, ResponseOutputMessage) for item in response_items)
        
        if not all_output_messages:
            logger.info("Not all items are ResponseOutputMessage, skipping extraction")
            return result
            
        # If we get here, all items are ResponseOutputMessage
        result["response_message_found"] = True
        
        # Extract text from the first ResponseOutputMessage (or concatenate if multiple)
        for item in response_items:
            logger.info(f"Processing ResponseOutputMessage with {len(item.content) if item.content else 0} content items")
            
            if item.content:
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        # If we already have text, append this text
                        if result["text"]:
                            result["text"] += "\n" + content_item.text
                        else:
                            result["text"] = content_item.text
                        logger.info(f"Extracted text (first 100 chars): {content_item.text[:100]}...")
        
        logger.info(f"Response message data extraction complete. Found: {result['response_message_found']}")
        return result

    def _send_search_results_to_frontend(self, web_search_data):
        """
        Send web search results to the frontend through agent_manager.
        
        Args:
            web_search_data: Dictionary containing extracted web search data
        """
        if not self.agent_manager:
            logger.warning("Agent manager not available, cannot send search results to frontend")
            return
            
        try:
            # Prepare the data in the format expected by the frontend
            search_result = {
                "text": web_search_data.get("text_with_citations", web_search_data.get("text", "")),
                "annotations": []
            }
            
            # Process annotations to match the frontend format
            if web_search_data.get("formatted_annotations"):
                for annotation in web_search_data.get("formatted_annotations", []):
                    if annotation.get("type") == "url_citation":
                        search_result["annotations"].append({
                            "title": annotation.get("title", ""),
                            "url": annotation.get("url", ""),
                            # Add snippet if available in the future
                        })
            
            # Log the search result for debugging
            logger.info(f"Prepared search result: {json.dumps(search_result)[:500]}...")
            
            # Convert the search result to a JSON string
            search_result_json = json.dumps(search_result)
            
            # Send the search result to the frontend
            logger.info("Sending search results to frontend")
            
            # Create a message in the format expected by the SearchAgentMessage component
            message = {
                "type": "agent",
                "name": "Search Agent",
                "content": {
                    "title": "Search Results",
                    "time": str(time.time()),
                    "image": "",
                    "description": search_result_json,
                    "action": "search",
                    "obs_time": "0",
                    "agent_time": "0",
                    "env_time": "0",
                    "token": 0,
                    "visualization": ""
                },
                "user_id": self.agent_manager.config.user_id,
            }
            
            # Determine which event to emit based on agent index
            event_name = 'message_response_left' if self.agent_manager.config.agent_idx == 0 else 'message_response_right'
            
            # Emit the message to the frontend
            self.agent_manager.config.socketio.emit(event_name, message)
            
            # Also update the session item in the conversation
            self.agent_manager.update_session_item({
                "role": "assistant",
                "type": "search_result",
                "content": search_result_json,
                "metadata": {
                    "annotations_count": len(search_result["annotations"]),
                    "timestamp": str(time.time())
                }
            })
            
            logger.info(f"Sent search results to frontend with {len(search_result['annotations'])} annotations")
        except Exception as e:
            logger.error(f"Error sending search results to frontend: {e}")

    def send_interact_message(self, text):
        """
        Send an interactive message from the agent to the user through agent_manager.
        
        This is used when the agent needs to ask questions or provide intermediate updates
        during processing, rather than just returning search results.
        
        Args:
            text: The text message to send to the user
        """
        if not self.agent_manager:
            logger.warning("Agent manager not available, cannot send interactive message")
            return
            
        try:
            # Log the message being sent
            logger.info(f"Sending interactive message to frontend: {text[:100]}...")
            
            # Call the agent_manager's send_interact_message method
            self.agent_manager.send_interact_message(text=text)
            
            logger.info("Interactive message sent to frontend successfully")
        except Exception as e:
            logger.error(f"Error sending interactive message to frontend: {e}")

    def _response_data_to_Arena_log(self, data, span_data: ResponseSpanData):
        """
        Convert ResponseSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The ResponseSpanData to convert (treat as read-only)

        Returns:
            Dictionary with ResponseSpanData fields mapped to Arena log format
        """
        data.span_name = "response"
        try:
            # Extract prompt messages from input if available
            if span_data.input:
                pass
            if span_data.response:
                response = span_data.response
                # Extract usage information if available
                if hasattr(response, "usage") and response.usage:
                    usage = span_data.response.usage
                    data.prompt_tokens = usage.input_tokens
                    data.completion_tokens = usage.output_tokens
                    data.total_request_tokens = usage.total_tokens

                # Extract model information if available
                if hasattr(response, "model"):
                    data.model = response.model

                # Extract completion message from response
                if hasattr(response, "output") and response.output:
                    response_items = response.output
                    
                    # Check if this is a web search response
                    web_search_data = self._extract_web_search_data(response_items)
                    
                    if web_search_data["web_search_found"]:
                        
                        # Store web search data
                        data.metadata = data.metadata or {}
                        data.metadata["web_search_id"] = web_search_data.get("web_search_id")
                        data.metadata["web_search_status"] = web_search_data.get("web_search_status")
                        data.metadata["web_search_type"] = web_search_data.get("web_search_type")
                        
                        # Store raw annotations
                        if web_search_data.get("annotations"):
                            data.metadata["annotations"] = web_search_data.get("annotations")
                        
                        # Store formatted annotations for better readability
                        if web_search_data.get("formatted_annotations"):
                            data.metadata["formatted_annotations"] = web_search_data.get("formatted_annotations")
                        
                        # Set the output text
                        data.output = web_search_data.get("text")
                        
                        # Log the extracted web search data
                        logger.info("Web search data extracted successfully:")
                        logger.info(f"Text (first 100 chars): {data.output[:100] if data.output else 'None'}...")
                        logger.info(f"Annotations count: {len(data.metadata.get('formatted_annotations', []))}")
                        
                        # Send the search results to the frontend
                        self._send_search_results_to_frontend(web_search_data)
                    
                    response_message_data = self._extract_response_message_data(response_items)
                    if response_message_data["response_message_found"]:
                        data.output = response_message_data["text"]
                        self.send_interact_message(text=response_message_data["text"])
                
            # Truncate base64 images in logs
            truncated_input = _truncate_base64_images(span_data.input)
            truncated_response = _truncate_base64_images(span_data.response)
                
            logger.info("===============Response data to Arena log===============")
            logger.info(truncated_input)
            logger.info(truncated_response)
            logger.info("===============End===============")
        except Exception as e:
            logger.error(f"Error converting response data to arena log: {e}")

    def _function_data_to_Arena_log(self, data, span_data: FunctionSpanData):
        """
        Convert FunctionSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The FunctionSpanData to convert (treat as read-only)

        Returns:
            Dictionary with FunctionSpanData fields mapped to Arena log format
        """
        try:
            data.span_name = span_data.name
            data.input = span_data.input
            data.output = span_data.output
            data.span_tools = [span_data.name]

            # Try to extract tool calls if the input is in a format that might contain them
            if span_data.input:
                data.log_type = "tool"
                data.input = span_data.input
                
            # Truncate base64 images in logs
            truncated_input = _truncate_base64_images(data.input)
            truncated_output = _truncate_base64_images(data.output)
                
            logger.info("===============Function data to Arena log===============")
            logger.info(truncated_input)
            logger.info(truncated_output)
            logger.info("===============End===============")
        except Exception as e:
            logger.error(f"Error converting function data to Arena log: {e}")

    def _generation_data_to_Arena_log(self, data, span_data: GenerationSpanData):
        """
        Convert GenerationSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The GenerationSpanData to convert (treat as read-only)

        Returns:
            Dictionary with GenerationSpanData fields mapped to Arena log format
        """
        data.span_name = "generation"
        data.model = span_data.model

        try:
            # Extract prompt messages from input if available
            if span_data.input:
                # Try to extract messages from input
                data.input = str(span_data.input)

            # Extract completion message from output if available
            if span_data.output:
                # Try to extract completion from output
                data.output = str(span_data.output)

            # Add model configuration if available
            if span_data.model_config:
                # Extract common LLM parameters from model_config
                for param in [
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                ]:
                    if param in span_data.model_config:
                        data[param] = span_data.model_config[param]

            # Add usage information if available
            if span_data.usage:
                data.prompt_tokens = span_data.usage.get("prompt_tokens")
                data.completion_tokens = span_data.usage.get("completion_tokens")
                data.total_request_tokens = span_data.usage.get("total_tokens")

            # Truncate base64 images in logs
            truncated_input = _truncate_base64_images(data.input)
            truncated_output = _truncate_base64_images(data.output)
                
            logger.info("===============Generation data to Arena log===============")
            logger.info(data.span_name)
            logger.info(data.model)
            logger.info(truncated_input)
            logger.info(truncated_output)
            logger.info(data.prompt_tokens)
            logger.info(data.completion_tokens)
            logger.info(data.total_request_tokens)
            logger.info("===============End===============")
        except Exception as e:
            logger.error(f"Error converting generation data to Arena log: {e}")

    def _handoff_data_to_Arena_log(self, data, span_data: HandoffSpanData):
        """
        Convert HandoffSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The HandoffSpanData to convert (treat as read-only)

        Returns:
            Dictionary with HandoffSpanData fields mapped to Arena log format
        """
        data.span_name = "handoff"
        data.span_handoffs = [f"{span_data.from_agent} -> {span_data.to_agent}"]
        data.metadata = {
            "from_agent": span_data.from_agent,
            "to_agent": span_data.to_agent,
        }

        logger.info("===============Handoff data to Arena log===============")
        logger.info(data.span_name)
        logger.info(data.span_handoffs)
        logger.info(data.metadata)
        logger.info("===============End===============")
        
        # Send handoff message to frontend if agent_manager is available
        if self.agent_manager:
            try:
                self.agent_manager.send_handoff_message(
                    from_agent=span_data.from_agent,
                    to_agent=span_data.to_agent
                )
                logger.info(f"Sent handoff notification to frontend: {span_data.from_agent} -> {span_data.to_agent}")
            except Exception as e:
                logger.error(f"Failed to send handoff notification to frontend: {e}")

    def _custom_data_to_Arena_log(self, data, span_data: CustomSpanData):
        """
        Convert CustomSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The CustomSpanData to convert (treat as read-only)

        Returns:
            Dictionary with CustomSpanData fields mapped to Arena log format
        """
        data.span_name = span_data.name
        # Create a deep copy of the data dictionary to avoid modifying the original
        data.metadata = copy.deepcopy(span_data.data) if span_data.data else {}

        # If the custom data contains specific fields that map to KeywordsAI fields, extract them
        for key in ["input", "output", "model", "prompt_tokens", "completion_tokens"]:
            if key in span_data.data:
                data[key] = copy.deepcopy(span_data.data[key])

        logger.info("===============Custom data to Arena log===============")
        logger.info(data)
        logger.info("===============End===============")

    def _agent_data_to_Arena_log(self, data, span_data: AgentSpanData):
        """
        Convert AgentSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The AgentSpanData to convert (treat as read-only)

        Returns:
            Dictionary with AgentSpanData fields mapped to Arena log format
        """
        data.span_name = span_data.name
        data.span_workflow_name = span_data.name

        # Add tools if available - make deep copies to avoid modifying originals
        if span_data.tools:
            data.span_tools = copy.deepcopy(span_data.tools)

        # Add handoffs if available - make deep copies to avoid modifying originals
        if span_data.handoffs:
            data.span_handoffs = copy.deepcopy(span_data.handoffs)

        # Add metadata with agent information
        data.metadata = {
            "output_type": span_data.output_type,
            "agent_name": span_data.name,
        }

        # Set metadata in log data
        data.metadata = data.metadata

        logger.info("===============Agent data to Arena log===============")
        logger.info(data.span_name)
        logger.info(data.span_workflow_name)
        logger.info(data.span_tools)
        logger.info(data.span_handoffs)
        logger.info(data.metadata)
        logger.info("===============End===============")

    def _guardrail_data_to_Arena_log(self, data, span_data: GuardrailSpanData):
        """
        Convert GuardrailSpanData to Arena log format.

        IMPORTANT: This function should ONLY READ from span_data and NEVER MODIFY it
        to avoid side effects in the agent context.

        Args:
            data: Base data dictionary with trace and span information
            span_data: The GuardrailSpanData to convert (treat as read-only)

        Returns:
            Dictionary with GuardrailSpanData fields mapped to Arena log format
        """
        data.span_name = f"guardrail:{span_data.name}"
        data.has_warnings = span_data.triggered
        if span_data.triggered:
            data.warnings_dict = data.warnings_dict or {}
            data.warnings_dict =  {
                f"guardrail:{span_data.name}": "guardrail triggered"
            }

        logger.info("===============Guardrail data to Arena log===============")
        logger.info(data.span_name)
        logger.info(data.has_warnings)
        logger.info(data.warnings_dict)
        logger.info("===============End===============")

    def _arena_export(
        self, item: Union[Trace, Span[Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Process different span types and extract all JSON serializable attributes.
        
        IMPORTANT: This function and its callees should NEVER modify the original span_data,
        only read from it and create new objects.

        Args:
            item: A Trace or Span object to export.

        Returns:
            A dictionary with all the JSON serializable attributes of the span,
            or None if the item cannot be exported.
        """
        # First try the native export method
        if isinstance(item, Trace):
            # This one is going to be the root trace. The span id will be the trace id
            return None  # We don't need the trace. Keywords AI will construct the trace from the spans
        elif isinstance(item, SpanImpl):
            # Get the span ID - it could be named span_id or id depending on the implementation

            # Create the base data dictionary with common fields
            data = SpanData(
                trace_unique_id=item.trace_id,
                span_unique_id=item.span_id,
                span_parent_id=item.parent_id,
                start_time=item.started_at,
                timestamp=item.ended_at,
                error_bit=1 if item.error else 0,
                status_code=400 if item.error else 200,
            )
            # data.latency = (data.timestamp - data.start_time).total_seconds()
            # Process the span data based on its type
            try:
                # Important: We do not modify the original span_data here, only read from it
                # and create new objects in the data variable
                if isinstance(item.span_data, ResponseSpanData):
                    self._response_data_to_Arena_log(data, item.span_data)
                elif isinstance(item.span_data, FunctionSpanData):
                    self._function_data_to_Arena_log(data, item.span_data)
                elif isinstance(item.span_data, GenerationSpanData):
                    self._generation_data_to_Arena_log(data, item.span_data)
                elif isinstance(item.span_data, HandoffSpanData):
                    self._handoff_data_to_Arena_log(data, item.span_data)
                elif isinstance(item.span_data, CustomSpanData):
                    self._custom_data_to_Arena_log(data, item.span_data)
                elif isinstance(item.span_data, AgentSpanData):
                    self._agent_data_to_Arena_log(data, item.span_data)
                elif isinstance(item.span_data, GuardrailSpanData):
                    self._guardrail_data_to_Arena_log(data, item.span_data)
                else:
                    logger.warning(f"Unknown span data type: {item.span_data}")
                    return None
                
                # Add to visualizer and display timeline
                result = data.model_dump(mode="json")
                visualizer = TraceVisualizer()
                visualizer.add_span(result)
                visualizer.visualize_latest_trace()
                
                return result
            except Exception as e:
                logger.error(
                    f"Error converting span data of {item.span_data} to Arena log: {e}"
                )
                return None
        else:
            return None

    def export(self, items: List[Union[Trace, Span[Any]]]) -> None:
        """
        Export traces and spans to the Keywords AI backend.

        Args:
            items: List of Trace or Span objects to export.
        """
        if not items:
            return

        if not self.api_key:
            logger.warning("API key is not set, skipping trace export")
            return

        # Process each item with our custom exporter
        data = [self._arena_export(item) for item in items]
        # Filter out None values
        data = [item for item in data if item]

        return data
        
        # TODO: no need to export to third party backend, only serve as a processor
        
        
        # if not data:
        #     return

        # payload = {"data": data}

        # headers = {
        #     "Authorization": f"Bearer {self.api_key}",
        #     "Content-Type": "application/json",
        #     "OpenAI-Beta": "traces=v1",
        # }

        # # Exponential backoff loop
        # attempt = 0
        # delay = self.base_delay
        # while True:
        #     attempt += 1
        #     try:
        #         response = self._client.post(
        #             url=self.endpoint, headers=headers, json=payload
        #         )

        #         # If the response is successful, break out of the loop
        #         if response.status_code < 300:
        #             logger.debug(f"Exported {len(data)} items to Keywords AI")
        #             return

        #         # If the response is a client error (4xx), we won't retry
        #         if 400 <= response.status_code < 500:
        #             logger.error(
        #                 f"Keywords AI client error {response.status_code}: {response.text}"
        #             )
        #             return

        #         # For 5xx or other unexpected codes, treat it as transient and retry
        #         logger.warning(f"Server error {response.status_code}, retrying.")
        #     except httpx.RequestError as exc:
        #         # Network or other I/O error, we'll retry
        #         logger.warning(f"Request failed: {exc}")

        #     # If we reach here, we need to retry or give up
        #     if attempt >= self.max_retries:
        #         logger.error("Max retries reached, giving up on this batch.")
        #         return

        #     # Exponential backoff + jitter
        #     sleep_time = delay + random.uniform(0, 0.1 * delay)  # 10% jitter
        #     time.sleep(sleep_time)
        #     delay = min(delay * 2, self.max_delay)


class ArenaTraceProcessor(BatchTraceProcessor):
    """
    A processor that uses ArenaSpanExporter to send traces and spans to Arena Backend.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        endpoint: str = "https://api.openai.com/v1/traces/ingest",
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        max_queue_size: int = 8192,
        max_batch_size: int = 128,
        schedule_delay: float = 0.5,
        export_trigger_ratio: float = 0.3,
        agent = None,
        agent_manager = None,
    ):
        """
        Initialize the Arena processor.

        Args:
            api_key: The API key for authentication.
            organization: The organization ID.
            project: The project ID.
            endpoint: The HTTP endpoint to which traces/spans are posted.
            max_retries: Maximum number of retries upon failures.
            base_delay: Base delay (in seconds) for the first backoff.
            max_delay: Maximum delay (in seconds) for backoff growth.
            max_queue_size: The maximum number of spans to store in the queue.
            max_batch_size: The maximum number of spans to export in a single batch.
            schedule_delay: The delay between checks for new spans to export.
            export_trigger_ratio: The ratio of the queue size at which we will trigger an export.
            agent: The agent instance associated with this processor.
            agent_manager: The agent manager instance for sending messages to frontend.
        """

        # Create the exporter
        exporter = ArenaSpanExporter(
            api_key=api_key,
            organization=organization,
            project=project,
            endpoint=endpoint,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            agent=agent,
            agent_manager=agent_manager,
        )

        # Initialize the BatchTraceProcessor with our exporter
        super().__init__(
            exporter=exporter,
            max_queue_size=max_queue_size,
            max_batch_size=max_batch_size,
            schedule_delay=schedule_delay,
            export_trigger_ratio=export_trigger_ratio,
        )

        # Store the exporter for easy access
        self._arena_exporter = exporter
        
        # Store agent and agent_manager for easy access
        self.agent = agent
        self.agent_manager = agent_manager
        
        # Log initialization
        if agent and agent_manager:
            logger.info(f"ArenaTraceProcessor initialized with agent and agent_manager")
        elif agent:
            logger.info(f"ArenaTraceProcessor initialized with agent only")
        elif agent_manager:
            logger.info(f"ArenaTraceProcessor initialized with agent_manager only")
        else:
            logger.info(f"ArenaTraceProcessor initialized without agent or agent_manager")


# Example usage:
"""
# Example: How to use the ArenaTraceProcessor with agent and agent_manager

from agents.tracing.processors import BatchTraceProcessor
from agents.tracing.traces import Trace
from agents.tracing.spans import Span

# Import your agent and agent manager classes
from your_agent_module import YourAgent
from agent_manager import AgentManager, SessionConfig

# Create your agent instance
agent = YourAgent(...)

# Create your agent manager instance
session_config = SessionConfig(
    user_id="user123",
    session_id="session456",
    region="us-east-1",
    agent_idx=0,
    session=None,
    conversation=None,
    socketio=socketio_instance,  # Your SocketIO instance
    stop_event=threading.Event()
)
agent_manager = AgentManager(agent, session_config)

# Create the trace processor with agent and agent_manager
processor = ArenaTraceProcessor(
    api_key="your_api_key",
    organization="your_org_id",
    project="your_project_id",
    agent=agent,
    agent_manager=agent_manager
)

# Now when handoff events occur, the agent_manager.send_handoff_message method 
# will be automatically called, and the frontend will receive the handoff notification
"""