# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # Load API key
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if not GOOGLE_API_KEY:
#     raise ValueError(" GOOGLE_API_KEY not found in .env file")

# # Load embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Load the FAISS index we saved
# print(" Loading FAISS index...")
# vectorstore = FAISS.load_local("storage/books_index", embeddings, allow_dangerous_deserialization=True)

# # Queries to test
# query = "Why Sports is necessary?"
# print(f"\n Query: {query}")

# # query = "What is Concept of Atomic Habbit?"
# # print(f"\n Query: {query}")


# # query = "What is The Great Gatsby about?"
# # print(f"\n Query: {query}")

# results = vectorstore.similarity_search(query, k=1)  
# print("\n Top Matches:")
# for i, res in enumerate(results, 1):
#     print(f"\n--- Match {i} ---")
#     print(f"(Length: {len(res.page_content)} characters)")
#     print(res.page_content)


  
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(" GOOGLE_API_KEY not found in .env file")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


print(" Loading FAISS index...")
vectorstore = FAISS.load_local("storage/books_index", embeddings, allow_dangerous_deserialization=True)


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Query
query = "What is the concept of Atomic Habits?"
print(f"\n Query: {query}")

# Search in FAISS
results = vectorstore.similarity_search_with_score(query, k=1)

if results:
    doc, score = results[0]
    print("\n Top Match (from FAISS):")
    print(f"(Score: {score:.4f})")
    print(doc.page_content[:400], "...\n")

    #threshold
    if score < 0.75:  
        print(" Low similarity → Falling back to Gemini LLM...\n")
        response = llm.invoke(query)
        print("💡 Gemini Answer:")
        print(response.content)
    else:
        print(" High similarity → Using RAG result only.")
else:
    print(" No matches found in FAISS. Falling back to Gemini...")
    response = llm.invoke(query)
    print(" Gemini Answer:")
    print(response.content)
