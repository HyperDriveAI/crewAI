import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import ExceptionTool
from langchain.agents.tools import InvalidTool
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping

from crewai.agents.cache.cache_hit import CacheHit
from crewai.tools.cache_tools import CacheTools
from crewai.utilities import I18N


class CrewAgentExecutor(AgentExecutor):
    i18n: I18N = I18N()
    iterations: int = 0
    request_within_rpm_limit: Any = None
    max_iterations: Optional[int] = 15
    force_answer_max_iterations: Optional[int] = None

    @root_validator()
    def set_force_answer_max_iterations(cls, values: Dict) -> Dict:
        """        Set the force answer maximum iterations.

        This function sets the 'force_answer_max_iterations' key in the input dictionary
        'values' to the value of 'max_iterations' minus 2.

        Args:
            cls: The class instance.
            values: A dictionary containing the input values.

        Returns:
            A dictionary with the updated 'force_answer_max_iterations' key.
        """

        values["force_answer_max_iterations"] = values["max_iterations"] - 2
        return values

    def _should_force_answer(self) -> bool:
        """        Check if the answer should be forced.

        Returns:
            bool: True if the number of iterations is equal to the maximum iterations for forcing the answer,
                False otherwise.
        """

        return True if self.iterations == self.force_answer_max_iterations else False

    def _force_answer(self, output: AgentAction):
        """        Force an answer with the given output.

        This method creates an AgentStep object with the provided output as the action and an error message
        obtained from self.i18n.errors("used_too_many_tools") as the observation.

        Args:
            output (AgentAction): The output to be used as the action in the AgentStep.

        Returns:
            AgentStep: An AgentStep object with the specified action and observation.
        """

        return AgentStep(
            action=output, observation=self.i18n.errors("used_too_many_tools")
        )

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """        Run text through and get agent response.

        This method runs the input text through the agent and returns the response. It constructs a mapping of tool names to tools for easy lookup, and a color mapping for logging. It then enters the agent loop until it returns a response, taking the next step and processing the output at each iteration.

        Args:
            inputs (Dict[str, str]): The input text to be processed.
            run_manager (Optional[CallbackManagerForChainRun]): An optional callback manager for chain run.

        Returns:
            Dict[str, Any]: A dictionary containing the agent response.
        """
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        self.iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(self.iterations, time_elapsed):
            if not self.request_within_rpm_limit or self.request_within_rpm_limit():
                next_step_output = self._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager=run_manager,
                )
                if isinstance(next_step_output, AgentFinish):
                    return self._return(
                        next_step_output, intermediate_steps, run_manager=run_manager
                    )

                intermediate_steps.extend(next_step_output)
                if len(next_step_output) == 1:
                    next_step_action = next_step_output[0]
                    # See if tool should return directly
                    tool_return = self._get_tool_return(next_step_action)
                    if tool_return is not None:
                        return self._return(
                            tool_return, intermediate_steps, run_manager=run_manager
                        )
                self.iterations += 1
                time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """        Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.

        Args:
            self: The instance of the class.
            name_to_tool_map (Dict[str, BaseTool]): A dictionary mapping tool names to their corresponding BaseTool objects.
            color_mapping (Dict[str, str]): A dictionary mapping tool names to their corresponding colors.
            inputs (Dict[str, str]): A dictionary containing input data for the agent.
            intermediate_steps (List[Tuple[AgentAction, str]]): A list of tuples containing AgentAction and observation strings.
            run_manager (Optional[CallbackManagerForChainRun]): An optional callback manager for chain runs.


        Yields:
            Iterator[Union[AgentFinish, AgentAction, AgentStep]]: Yields instances of AgentFinish, AgentAction, or AgentStep.

        Raises:
            ValueError: If an unexpected output type is received from the agent.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
            if self._should_force_answer():
                if isinstance(output, AgentAction):
                    output = output
                elif isinstance(output, CacheHit):
                    output = output.action
                else:
                    raise ValueError(
                        f"Unexpected output type from agent: {type(output)}"
                    )
                yield self._force_answer(output)
                return

        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )

            if self._should_force_answer():
                yield self._force_answer(output)
                return

            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        # Override tool usage to use CacheTools
        if isinstance(output, CacheHit):
            cache = output.cache
            action = output.action
            tool = CacheTools(cache_handler=cache).tool()
            output = action.copy()
            output.tool_input = f"tool:{action.tool}|input:{action.tool_input}"
            output.tool = tool.name
            name_to_tool_map[tool.name] = tool
            color_mapping[tool.name] = color_mapping[action.tool]

        actions: List[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        yield from actions
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            yield AgentStep(action=agent_action, observation=observation)
