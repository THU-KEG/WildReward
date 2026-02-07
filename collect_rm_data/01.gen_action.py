import json
import re
from pathlib import Path

def load_conversations(path):
    """Load conversations from a JSON file.
    Each conversation should be a list of turns with 'role' and 'content'."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
        return data

def format_history(history):
    """Format conversation turns for readability."""
    formatted = []
    for turn in history:
        role = turn["role"].capitalize()
        content = turn["content"].strip().replace("\n", " ")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)

# work
def build_preference_prompt(messages, user_query):
    """
    Build a structured prompt that asks the model to classify
    the user's preference toward the previous assistant response.
    """
    prompt = f"""
You are an expert annotator. Your task is to infer how satisfied the user was 
with the assistant’s previous response, based solely on the user’s latest message.

IMPORTANT RULES:
- Only use strong and explicit evidence to classify satisfaction.
- Do NOT assume the user is satisfied or inspired just because they continue the topic.
- Users often ask follow-up questions even when they are dissatisfied.
- Neutral, ambiguous, or topic-extending queries should NOT be labeled as "inspired".

<User's previous message>:
{messages[0]['content']}

<Assistant's previous response>:
{messages[1]['content']}

<User's latest message>:
{user_query}

Based on the user's latest message, classify their preference toward the assistant's previous response into one of the following categories:

--------------------------------------
CATEGORIES (strong evidence only)
--------------------------------------

[[1]] CLEARLY NEGATIVE / REJECTION  
User explicitly criticizes, rejects, or expresses frustration.

Examples:
- “This is wrong.”
- “You didn’t answer my question.”
- “No, that’s not what I need.”

--------------------------------------------------

[[2]] CORRECTION / ERROR POINTER (Negative)
User points out a **mistake**, **missing constraint**, or **hallucination** in the previous response.
The assistant failed to follow the original instruction perfectly.

Examples:
- "You calculated the last step wrong."
- "I asked for Python, not C++."
- "You forgot to mention the limitations."
- "This code doesn't run."

--------------------------------------------------

[[3]] NEUTRAL / UNCLEAR
User shows no clear positive/negative attitude.  
Question is unrelated, generic, or ambiguous.  
May simply continue asking questions without emotional signals.

Examples:
- “Okay, next question.”
- “What is the formula for X?”
- “How does this apply to Y?” (no emotional cue)

IF THE MESSAGE IS UNCLEAR (CATEGORY 3):
- Optionally, check the assistant response quality.
- If the response is objectively high-quality (correct, helpful, structured), mark as 4.
- If average, leave as 3.
- If poor (misleading, wrong, unsafe), mark as 1.
- DO NOT mark as 5 (that requires explicit user satisfaction).

Examples:
- “Okay, next question.” → 3 or fallback 4/1 based on response quality

--------------------------------------------------

[[4]] POSITIVE ENGAGEMENT (strong evidence only)
User explicitly builds upon the response **with positive emotional framing**  
(e.g., excitement, interest, approval). Not just continuing the topic.

Examples:
- “Interesting, then what happens if we scale it?”
- “That’s helpful — can we apply it to drones?”
- “Great point! What about the dynamic case?”

DO NOT label as 4 unless there is **clear positive emotion + meaningful extension**.

--------------------------------------------------

[[5]] CLEAR SATISFACTION
User expresses gratitude, satisfaction, or says the problem is solved.

Examples:
- “Thanks, this solves it.”
- “Perfect answer.”
- “That’s all I need.”

--------------------------------------
OUTPUT FORMAT
--------------------------------------
[[<category_number>]] <brief_reasoning>

"""
    return prompt.strip()



def parse_preference_output(output_text):
    """
    Parse the model output to extract the category number (1-5)
    and reasoning if available.
    Example input: '[[3]] The user changes topic with no emotional signal.'
    Returns: {'category': 3, 'reason': 'The user changes topic with no emotional signal.'}
    """
    match = re.search(r"\[\[(\d)\]\]\s*(.*)", output_text.strip())
    if match:
        category = int(match.group(1))
        reason = match.group(2).strip()
        return {"category": category, "reason": reason}
    else:
        return {"category": None, "reason": output_text.strip()}


def main(conversation_path, output_path, parse_mode=False):
    """
    Main entry: generate preference classification prompts.
    Optionally parse model outputs if `parse_mode` is True.
    """
    conversations = load_conversations(conversation_path)
    results = []

    for conv in conversations:
        prompt = build_preference_prompt(conv["messages"], conv["user_query"])
        if prompt:
            conv["prompt"] = prompt
            results.append(conv)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item)+"\n")
        # json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(results)} preference prompts and saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate structured preference classification prompts from conversation history."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input conversations JSON file."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save generated prompts JSON file."
    )
    parser.add_argument(
        "--parse", action="store_true",
        help="If provided, will attempt to parse model outputs using [[<n>]] format."
    )
    parser.add_argument(
        "--example-output", type=str, default=None,
        help="Optional path to model output file to parse (only used with --parse)."
    )

    args = parser.parse_args()

    # Default behavior: generate prompts
    if not args.parse:
        main(args.input, args.output)
    else:
        # If parse mode: parse model outputs (for evaluation/debug)
        if not args.example_output:
            raise ValueError("You must provide --example-output file when using --parse.")
        with open(args.example_output, 'r', encoding='utf-8') as f:
            outputs = json.load(f)
        parsed = [parse_preference_output(o.get("output", "")) for o in outputs]
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
