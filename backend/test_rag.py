from rag_pipeline import rag_pipeline

print("🚀 Starting test...")

query = "What is deadlock?"

result = rag_pipeline(query)

print("\n✅ ANSWER:\n")
print(result["answer"])

print("\n📚 SOURCES:\n")
for s in result["sources"]:
    print("-", s[:200], "\n")