from types import SimpleNamespace


def text_response(text):
    return SimpleNamespace(
        output=[SimpleNamespace(type="message")],
        output_text=text,
    )


def tool_call_response(arguments, name="search_study_materials", call_id="call_1"):
    return SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(
                type="function_call",
                name=name,
                call_id=call_id,
                arguments=arguments,
            ),
        ],
        output_text="",
    )


def stream(*deltas):
    events = [SimpleNamespace(type="response.created")]
    events += [SimpleNamespace(type="response.output_text.delta", delta=d) for d in deltas]
    events.append(SimpleNamespace(type="response.completed"))
    return iter(events)


class FakeResponses:
    """Returns queued results in order and records every create() call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)
