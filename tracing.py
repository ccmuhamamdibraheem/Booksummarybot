from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure the tracer
resource = Resource(attributes={"service.name": "book-rag-chatbot"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Export traces to Jaeger (via OTLP)
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Jaeger OTLP gRPC
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

print("Tracing initialized with OTLP → Jaeger 🚀")
print("checking how git and github works")
print("checking how git and github works 2")
print("checking how new branch thing works")