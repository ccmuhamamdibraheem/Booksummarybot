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
st.write("Paste a line or short paragraph and I will try to find which book it comes from.")

# query = st.text_input("Paste the line here:", key="source_detect_input")

uploaded_file = st.file_uploader("Upload your questions file (.txt)", type=["txt"])

# ------------------- SETTINGS -------------------
SIMILARITY_THRESHOLD = 0.40
TOP_K = 3

def fmt_score(s):
    try:
        return f"{s:.4f}"
    except Exception:
        return str(s)

# ------------------- QUERY HANDLING -------------------
def estimate_tokens(text):
    # Rough estimate: 1 token ~ 4 characters in English
    return max(1, len(text) // 4)

if query:
    with tracer.start_as_current_span("user_query_received") as user_span:
        user_span.set_attribute("query.text", query)
        st.write(f"### 🔍 Searching for source of the line:")

        # GENERATE EMBEDDING
        with tracer.start_as_current_span("generate_embedding") as emb_span:
            emb_model = getattr(embeddings, "model", "unknown")
            import time
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

            with tracer.start_as_current_span("render_results") as render_span:
                render_span.set_attribute("response.length", len(doc.page_content))
                render_span.set_attribute("used_llm", False)
                st.success(f"Matched book: **{title}** — *{author}* (score: {fmt_score(score)})")
                st.write("---")
                st.write(doc.page_content)

                # Show other candidates
                if len(results) > 1:
                    if st.checkbox("Show other top matches", key="show_other_matches"):
                        st.write("### Other candidates:")
                        for i, (cand_doc, cand_score) in enumerate(results[1:], start=2):
                            t = cand_doc.metadata.get("title", "Unknown Title")
                            a = cand_doc.metadata.get("author", "Unknown Author")
                            p = cand_doc.metadata.get("page", "Unknown")
                            st.write(f"**{i}. {t}** — {a} (score: {fmt_score(cand_score)})")
                            st.write(cand_doc.page_content[:800])
                            st.write("---")

        # IF NO GOOD MATCH
        else:
            st.warning("I couldn't find a confident match in the database.")
            st.write("Would you like me to try searching with the LLM (may use external knowledge)?")

            use_llm = st.radio("Search with LLM?", ("No", "Yes"), index=0, key="use_llm_radio")

            if use_llm == "Yes":
                with tracer.start_as_current_span("llm_api_call") as llm_span:
                    llm_span.set_attribute("llm.model", getattr(llm, "model", "unknown"))
                    llm_span.set_attribute("llm.endpoint", "invoke")
                    import time
                    t0 = time.time()
                    if not results:
                        response = llm.invoke(query)
                        prompt_used = query
                    else:
                        context = "\n\n".join([d.page_content for d, s in results])
                        prompt_used = (
                            "You are given some context from books and a user line. "
                            "If the line matches a book, say which book (title and author) and give a short excerpt. "
                            "If it does not match any book, just answer the user's line directly using your own knowledge, "
                            "without saying MATCHED or EXCERPT.\n\n"
                            f"CONTEXT:\n{context}\n\nLINE:\n{query}\n\n"
                            "Respond appropriately."
                        )
                        response = llm.invoke(prompt_used)
                    # Extract content from response
                    if hasattr(response, "content"):
                        content = response.content
                    else:
                        content = str(response)
                    latency = int((time.time() - t0) * 1000)
                    llm_span.set_attribute("llm.latency_ms", latency)
                    # If Gemini API returns token usage, add here:
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

                user_span.set_attribute("used_llm", True)