prompt = f"""
  Analyze the following user query and generate two components:
  1. **Purpose**: Identify the specific task or objective the user wants the LLM to accomplish.
  2. **Information**: Extract all contextual details, data, assumptions, and background provided by the user (even if labeled with terms like "assume").

  **Definitions**:
  - **Purpose**: The explicit or implicit task the user expects the LLM to solve (e.g., calculation, analysis, generation, explanation).
  - **Information**: All supporting details, including numerical values, descriptions, assumptions, or contextual facts (e.g., "Assume X = 5%" is treated as information).

  **Example**:
  - **Query**: "Calculate the monthly payment for a $200,000 loan over 30 years. Assume the interest rate is 5%."
  - **Purpose**: "Calculate the monthly payment for a loan."
  - **Information**: ["Loan amount: $200,000", "Loan term: 30 years", "Assumed interest rate: 5%"].

  **Instructions**:
  1. Read the user’s query carefully.
  2. Separate the **task** (purpose) from **supporting details** (information).
  3. Return the result in JSON format:
  {{
    "purpose": "[concise task description]",
    "information": ["list", "of", "extracted", "details"]
  }}

  **User Query**:
  "{query}"
  """