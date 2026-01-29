import streamlit as st

# Page configuration
st.set_page_config(
    page_title="IntelliVault",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 IntelliVault")
st.caption("Your Personal Knowledge Vault")

st.divider()

# --Upload Section--
st.header("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Upload"):
    if uploaded_files:
        for file in uploaded_files:
            st.success(f"Uploaded: {file.name}")
    else:
        st.warning("Please upload at least one file.")

if st.button("Index Documents"):
    st.success("Documents indexed successfully (placeholder).")

st.divider()

# --Q&A Section--
st.header("Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        st.subheader("Answer")
        st.write("This is a placeholder answer generated from your documents.")

        st.subheader("Source Document")
        st.write("example_notes.pdf")

st.divider()

# Footer
st.caption("Backend integration coming in the next phase")
