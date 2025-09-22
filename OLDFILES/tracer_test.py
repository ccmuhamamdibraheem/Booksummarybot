from tracing import tracer

with tracer.start_as_current_span("test_span"):
    print("Tracing test!")