import pytest
from unittest.mock import patch, ANY, AsyncMock # Removed MagicMock as it's in conftest
import importlib # Import importlib for reloading

# Removed local mock definitions for StateGraph
# They will be provided by fixtures from conftest.py

# Mock for the planner_agent function (if needed for specific tests, otherwise remove)
# mock_planner_agent_func = AsyncMock(name="mock_run_planner_agent_func") # Keep if used, otherwise remove

# Import the correct InsightFlowState for type checking in assertions
from insight_state import InsightFlowState

@patch('langgraph.graph.StateGraph') # Patched at source
def test_langgraph_basic_setup(mock_state_graph_class_actual):
    """Test that StateGraph is initialized with the correct state type and entry point is set."""
    import app 
    importlib.reload(app) # Force reload app module AFTER mocks are applied
    # The InsightFlowState used in app.StateGraph(InsightFlowState) should be the one from state.py
    
    # Check that StateGraph was called with InsightFlowState at least once (due to reload)
    mock_state_graph_class_actual.assert_any_call(InsightFlowState)
    
    # Get the instance from the last call (which is the one relevant to the reloaded app module)
    mock_sg_instance = mock_state_graph_class_actual.return_value
    # set_entry_point will also be called multiple times due to reload
    mock_sg_instance.set_entry_point.assert_any_call("planner_agent")


@patch('langgraph.graph.StateGraph') # Patched at source
def test_langgraph_adds_planner_node(mock_state_graph_class_actual):
    """Test that the planner_agent node is added to the graph."""
    import app 
    importlib.reload(app) 
    mock_sg_instance = mock_state_graph_class_actual.return_value
    # Assert that add_node was called with "planner_agent" and some callable
    # We can get more specific by checking the actual function if app.run_planner_agent is accessible
    mock_sg_instance.add_node.assert_any_call("planner_agent", app.run_planner_agent)


@patch('langgraph.graph.StateGraph') # Patched at source
def test_langgraph_adds_execute_persona_tasks_node(mock_state_graph_class_actual):
    """Test that the execute_persona_tasks node is added."""
    import app 
    importlib.reload(app) 
    mock_sg_instance = mock_state_graph_class_actual.return_value
    mock_sg_instance.add_node.assert_any_call("execute_persona_tasks", app.execute_persona_tasks)


@patch('langgraph.graph.StateGraph') # Patched at source
def test_langgraph_adds_synthesize_responses_node(mock_state_graph_class_actual):
    """Test that the synthesize_responses node is added."""
    import app 
    importlib.reload(app) 
    mock_sg_instance = mock_state_graph_class_actual.return_value
    mock_sg_instance.add_node.assert_any_call("synthesize_responses", app.synthesize_responses)


@patch('langgraph.graph.StateGraph') # Patched at source
def test_langgraph_adds_generate_visualization_node(mock_state_graph_class_actual):
    """Test that the generate_visualization node is added."""
    import app 
    importlib.reload(app) 
    mock_sg_instance = mock_state_graph_class_actual.return_value
    mock_sg_instance.add_node.assert_any_call("generate_visualization", app.generate_visualization)


@patch('langgraph.graph.StateGraph') # Patched at source
def test_langgraph_adds_present_results_node(mock_state_graph_class_actual):
    """Test that the present_results node is added."""
    import app
    importlib.reload(app)
    mock_sg_instance = mock_state_graph_class_actual.return_value
    mock_sg_instance.add_node.assert_any_call("present_results", app.present_results)