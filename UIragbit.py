import asyncio
import nest_asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from tracing import tracer  

# ------------------- ASYNCIO FIX -------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

# ------------------- LOAD API KEY -------------------
with tracer.start_as_current_span("app_startup"):
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found in .env file")
        st.stop()

# ------------------- LOAD EMBEDDINGS & INDEX -------------------
with tracer.start_as_current_span("init_embeddings"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

with tracer.start_as_current_span("load_faiss_index"):
    try:
        vectorstore = FAISS.load_local(
            "storage/books_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        with tracer.start_as_current_span("error") as error_span:
            error_span.set_attribute("error.type", type(e).__name__)
            error_span.set_attribute("error.message", str(e))
        st.error(f"Failed to load FAISS index: {e}")
        st.stop()

# ------------------- INITIALIZE LLM -------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=1.5)

# ------------------- STREAMLIT UI -------------------
st.title("Book RAG Assistant — Source Detector")
st.write("Upload a text file with one question per line. I will process each question and show detailed stats.")

uploaded_file = st.file_uploader("Upload your questions file (.txt)", type=["txt"])

# ------------------- SETTINGS -------------------
SIMILARITY_THRESHOLD = 0.40
TOP_K = 3

def fmt_score(s):
    try:
        return f"{s:.4f}"
    except Exception:
        return str(s)

def estimate_tokens(text):
    # Rough estimate: 1 token ~ 4 characters in English
    return max(1, len(text) // 4)

if uploaded_file:
    # Read questions from uploaded file
    questions = [line.strip() for line in uploaded_file.readlines() if line.strip()]
    # Decode bytes to string if needed
    questions = [q.decode("utf-8") if isinstance(q, bytes) else q for q in questions]

    import time
    overall_start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    results_summary = []

    with tracer.start_as_current_span("multi_query_session") as session_span:
        for idx, query in enumerate(questions, 1):
            q_start = time.time()
            used_llm = False
            with tracer.start_as_current_span("user_query_received") as user_span:
                user_span.set_attribute("query.text", query)
                st.write(f"### 🔍 Q{idx}: {query}")

                # GENERATE EMBEDDING
                with tracer.start_as_current_span("generate_embedding") as emb_span:
                    emb_model = getattr(embeddings, "model", "unknown")
                    t0 = time.time()
                    query_emb = embeddings.embed_query(query)
                    emb_span.set_attribute("embedding.model", emb_model)
                    emb_span.set_attribute("embedding.length", len(query_emb))
                    emb_span.set_attribute("embedding.latency_ms", int((time.time() - t0) * 1000))

                # VECTOR SEARCH
                with tracer.start_as_current_span("vector_search") as vs_span:
                    t0 = time.time()
                    results = vectorstore.similarity_search_with_score(query, k=TOP_K)
                    vs_span.set_attribute("num_results", len(results))
                    if results:
                        top_doc, top_score = results[0]
                        vs_span.set_attribute("top_score", float(top_score))
                        is_relevant = top_score <= SIMILARITY_THRESHOLD
                    else:
                        top_doc, top_score = None, None
                        is_relevant = False
                    vs_span.set_attribute("vector_search.latency_ms", int((time.time() - t0) * 1000))

                    # EVALUATE MATCH
                    with tracer.start_as_current_span("evaluate_match") as eval_span:
                        eval_span.set_attribute("top_score", float(top_score) if top_score else -1)
                        eval_span.set_attribute("threshold", SIMILARITY_THRESHOLD)
                        eval_span.set_attribute("is_relevant", is_relevant)

                # IF DATABASE MATCH
                if is_relevant:
                    doc, score = results[0]
                    title = doc.metadata.get("title", "Unknown Title")
                    author = doc.metadata.get("author", "Unknown Author")
                    page = doc.metadata.get("page", "Unknown")
                    user_span.set_attribute("book_title", title)
                    user_span.set_attribute("book_author", author)
                    user_span.set_attribute("page_number", page)
                    answer_source = "database"
                    prompt_used = query
                    content = doc.page_content
                    with tracer.start_as_current_span("render_results") as render_span:
                        render_span.set_attribute("response.length", len(doc.page_content))
                        render_span.set_attribute("used_llm", False)
                        st.success(f"Matched book: **{title}** — *{author}* (score: {fmt_score(score)})")
                        st.write("---")
                        st.write(doc.page_content)
                # IF NO GOOD MATCH
                else:
                    answer_source = "llm"
                    st.warning("I couldn't find a confident match in the database.")
                    with tracer.start_as_current_span("llm_api_call") as llm_span:
                        llm_span.set_attribute("llm.model", getattr(llm, "model", "unknown"))
                        llm_span.set_attribute("llm.endpoint", "invoke")
                        t0 = time.time()
                        prompt_used = query
                        response = llm.invoke(query)
                        if hasattr(response, "content"):
                            content = response.content
                        else:
                            content = str(response)
                        latency = int((time.time() - t0) * 1000)
                        llm_span.set_attribute("llm.latency_ms", latency)
                        input_tokens = estimate_tokens(prompt_used)
                        output_tokens = estimate_tokens(content)
                        llm_span.set_attribute("llm.tokens.input_est", input_tokens)
                        llm_span.set_attribute("llm.tokens.output_est", output_tokens)
                        llm_span.set_attribute("llm.tokens.total_est", input_tokens + output_tokens)
                        llm_span.set_attribute("llm.status", "success")
                        with tracer.start_as_current_span("render_results") as render_span:
                            render_span.set_attribute("response.length", len(content))
                            render_span.set_attribute("used_llm", True)
                            st.info("🤖 LLM Response")
                            st.write(content)
                    used_llm = True

                # Token counting for both cases
                input_tokens = estimate_tokens(prompt_used)
                output_tokens = estimate_tokens(content)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                q_time = time.time() - q_start

                user_span.set_attribute("answer_source", answer_source)
                user_span.set_attribute("latency_sec", q_time)
                user_span.set_attribute("input_tokens", input_tokens)
                user_span.set_attribute("output_tokens", output_tokens)
                user_span.set_attribute("total_tokens", input_tokens + output_tokens)
                user_span.set_attribute("used_llm", used_llm)

                results_summary.append({
                    "question": query,
                    "source": answer_source,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "latency": q_time,
                })

        overall_time = time.time() - overall_start
        session_span.set_attribute("total_input_tokens", total_input_tokens)
        session_span.set_attribute("total_output_tokens", total_output_tokens)
        session_span.set_attribute("total_tokens", total_input_tokens + total_output_tokens)
        session_span.set_attribute("overall_time_sec", overall_time)

        # Show summary table in Streamlit
        st.write(f"**Total tokens used:** {total_input_tokens + total_output_tokens}")
        st.write(f"**Total time taken:** {overall_time:.2f} seconds")

        # Bottleneck detection
        if results_summary:
            slowest = max(results_summary, key=lambda r: r["latency"])
            st.warning(f"Bottleneck: Q: '{slowest['question']}' took the longest ({slowest['latency']:.2f} seconds)")