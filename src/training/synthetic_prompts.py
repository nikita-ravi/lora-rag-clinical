"""Prompt templates for synthetic data generation.

These prompts are used to generate LoRA-B training targets via Claude Haiku.
"""

SYNTHETIC_GENERATION_PROMPT = """You are a medical expert. Given a question and evidence passages, provide a reasoned answer that explicitly cites the evidence.

CRITICAL REQUIREMENTS:
1. Every factual claim must reference a specific passage by index (e.g., "[1]", "[3]")
2. Your final answer must be exactly: {gold_answer}
3. Your reasoning must directly support this answer using the evidence

Question: {question}

Evidence:
{passages}

Respond in this exact format:
Based on the provided evidence, [your reasoning with citations like [1], [2], etc.]

Answer: {gold_answer}

Your response:"""

# Alternative prompt for when model struggles with citations
SYNTHETIC_GENERATION_PROMPT_STRICT = """You are a medical expert. Given a question and numbered evidence passages, provide a reasoned answer.

STRICT REQUIREMENTS:
1. You MUST cite passages using [1], [2], etc. format
2. Every sentence with a factual claim needs at least one citation
3. Your final answer MUST be exactly: {gold_answer}
4. Do not add any caveats or qualifications to the answer

Question: {question}

Evidence:
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}
[4] {passage_4}
[5] {passage_5}

Format your response as:
Based on the provided evidence, [reasoning with [1], [2], etc. citations]

Answer: {gold_answer}"""


def format_generation_prompt(
    question: str,
    passages: list[dict],
    gold_answer: str,
    strict: bool = False,
) -> str:
    """Format the synthetic generation prompt.

    Args:
        question: The question text
        passages: List of passage dicts with "text" key
        gold_answer: The gold answer (yes/no/maybe or entity)
        strict: Whether to use the stricter prompt variant

    Returns:
        Formatted prompt string
    """
    raise NotImplementedError("TODO: Implement in M4")


def get_prompt_hash() -> str:
    """Get hash of the generation prompt for reproducibility tracking."""
    raise NotImplementedError("TODO: Implement in M4")
