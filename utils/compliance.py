import google.generativeai as genai

def check_rule(rule, index, vectors, chunks, embedder):
    retrieved_chunks = retrieve(rule["description"], index, vectors, chunks, embedder)
    context = "\n---\n".join(retrieved_chunks)

    prompt = f"""
You are a compliance-checking assistant.

Rule: {rule['name']}
Description: {rule['description']}

Context:
{context}

Respond in the following structure:
1. Is the rule satisfied? (Yes/No)
2. Evidence found in the contract.
3. Remediation steps if non-compliant.
"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    return response.text
