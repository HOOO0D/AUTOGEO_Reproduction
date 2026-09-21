print("[1] Program started")

from autogeo.rewriters import rewrite_document

print("[2] AutoGEO imported")

document = """
AutoGEO automatically extracts content preference rules from
generative engines and rewrites documents to maximize visibility
while preserving accuracy.
"""

print("[3] Calling rewrite_document")

rewritten_text = rewrite_document(
    document=document,
    dataset="Researchy-GEO",
    engine_llm="gemini"
)

print("[4] Rewrite completed")

print(rewritten_text)