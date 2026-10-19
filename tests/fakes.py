from types import SimpleNamespace


def function_call(arguments, name="search_study_materials", call_id="call_1"):
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=arguments)


def text_response(text):
    return SimpleNamespace(
        output=[SimpleNamespace(type="message")],
        output_text=text,
    )


def tool_call_response(*calls):
    return SimpleNamespace(
        output=[SimpleNamespace(type="reasoning"), *calls],
        output_text="",
    )


def stream(*deltas, calls=()):
    """A Responses API event stream: text deltas, then any function calls the model makes."""
    events = [SimpleNamespace(type="response.created")]
    events += [SimpleNamespace(type="response.output_text.delta", delta=d) for d in deltas]
    events += [SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(type="message"))]
    events += [SimpleNamespace(type="response.output_item.done", item=c) for c in calls]
    events.append(SimpleNamespace(type="response.completed"))
    return iter(events)


class FakeResponses:
    """Returns queued results in order and records every create() call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
