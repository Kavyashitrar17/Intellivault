"""
prompts.py
----------
Centralizes all text templates used in the system.

WHY A SEPARATE FILE?
  Keeps your logic files clean. If you want to change how responses
  are worded, you only edit this one file.

  Also useful when demoing your viva — you can show "prompt engineering"
  as a deliberate design decision.

CHANGES FROM ORIGINAL:
  - Was empty. Now populated with templates used by qa_chain.py and api.py.
"""


# -------------------------------------------------------
# Shown when no documents have been uploaded yet
# -------------------------------------------------------
NO_DOCUMENTS_MSG = (
    "No documents have been uploaded yet. "
    "Please upload a PDF or TXT file first using the Upload section."
)

# -------------------------------------------------------
# Shown when a query returns no results above the threshold
# -------------------------------------------------------
NO_RESULTS_MSG = (
    "I couldn't find a relevant answer in your documents for that question. "
    "Try rephrasing, or check that the right document has been uploaded."
)

# -------------------------------------------------------
# Shown when a file is empty or unreadable
# -------------------------------------------------------
EMPTY_FILE_MSG = (
    "The uploaded file appears to be empty or could not be read. "
    "Please check the file and try again."
)

# -------------------------------------------------------
# Used to format the final answer with a source attribution line
# -------------------------------------------------------
def format_answer_with_source(answer: str, source_files: list) -> str:
    """
    Append a source line to the answer for transparency.

    Example output:
        "A deadlock occurs when two processes wait for each other indefinitely.
         Source: os_notes.pdf"
    """
    if not source_files:
        return answer

    sources_str = ", ".join(sorted(set(source_files)))
    return f"{answer}\n\n📄 Source: {sources_str}"