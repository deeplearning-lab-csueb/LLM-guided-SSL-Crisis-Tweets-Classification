import re

#-----------------------------------FOR MODIFIED RULES (1 AND 2) -----------------------------------------#
LABELS = [
  "caution_and_advice",
  "displaced_people_and_evacuations",
  "infrastructure_and_utility_damage",
  "injured_or_dead_people",
  "missing_or_found_people",
  "requests_or_urgent_needs",
  "rescue_volunteering_or_donation_effort",
  "sympathy_and_support",
  "other_relevant_information",  
  "not_humanitarian",
]

SYSTEM_PROMPT = (
  "You are a precise tweet classifier for humanitarian-response content. "
  "Choose exactly one label from the allowed labels that best fits the tweet. "
  "If unrelated to humanitarian contexts, choose 'not_humanitarian'. "
  "Follow the short rules and output JSON that matches the schema; no extra fields."
)

def filter_rules_by_labels(rules_text: str, allowed_labels: list[str]) -> str:
    """
    Keep only bullet lines that define labels we actually allow for this event.
    Lines that don't start with a label definition (e.g., tie-break notes) are kept.
      Expected bullet format: "- label_name: description..."
    """
    allowed = set(str(x).strip() for x in allowed_labels)
    out_lines = []
    for raw in rules_text.splitlines():
        line = raw.rstrip()
        m = re.match(r'^\s*-\s*([a-z0-9_]+)\s*:\s*(.+)$', line, flags=re.I)
        if not m:
            # keep notes like "Tie-break: ..." or blank lines
            out_lines.append(line)
            continue
        label = m.group(1).strip()
        if label in allowed:
            out_lines.append(line)
    return "\n".join(out_lines).strip()

def make_user_message(tweet_text: str, rules_text: str, current_labels: list[str]) -> str:
    """
    Build the user message with per-event filtered rules and the explicit allowed list.
    """
    rules_filtered = filter_rules_by_labels(rules_text, current_labels)
    allowed_csv = ", ".join(current_labels)

    return (
        "Classify this tweet into exactly ONE of the allowed labels.\n"
        f"Allowed labels (enum): {allowed_csv}\n\n"
        "Rules:\n"
        f"{rules_filtered}\n\n"
        "Return JSON that conforms to the provided schema.\n"
        f'Tweet: """{tweet_text}"""'
    )
    
#-----------------------------------FOR MODIFIED RULES (1 AND 2) -----------------------------------------#


#-----------------------------------FOR RULES DF (ORIGINAL RULES FROM IMRAN) -----------------------------------------#
# LABELS = [
#     "caution_and_advice",
#     "sympathy_and_support",
#     "requests_or_urgent_needs",
#     "displaced_people_and_evacuations",
#     "injured_or_dead_people",
#     "missing_or_found_people",
#     "infrastructure_and_utility_damage",
#     "rescue_volunteering_or_donation_effort",
#     "other_relevant_information",
#     "not_humanitarian"
# ]

# SYSTEM_PROMPT = """Read the category names and their definitions below, then classify the following tweet into the appropriate category. In your response, mention only the category name.
# Category name: category definition
# - caution_and_advice: Reports of warnings issued or lifted, guidance and tips related to the disaster.
# - sympathy_and_support: Tweets with prayers, thoughts, and emotional support.
# - requests_or_urgent_needs: Reports of urgent needs or supplies such as food, water, clothing, money,...
# - displaced_people_and_evacuations: People who have relocated due to the crisis, even for a short time...
# - injured_or_dead_people: Reports of injured or dead people due to the disaster.
# - missing_or_found_people: Reports of missing or found people due to the disaster.
# - infrastructure_and_utility_damage: Reports of any type of damage to infrastructure such as buildings, houses,...
# - rescue_volunteering_or_donation_effort: Reports of any type of rescue, volunteering, or donation efforts...
# - not_humanitarian: If the tweet does not convey humanitarian aid-related information.
# """

# def make_user_message(tweet_text: str, rules: str, labels: list[str] = LABELS) -> str:
#     return (
#         f"Category: {labels}\n"
#         "Rules:\n"
#         f"{rules.strip()}\n"
#         "Choose exactly one category that best matches the Tweet below.\n"
#         f'Tweet: """{tweet_text}"""'
#         f"Category: "
#     )
#-----------------------------------FOR RULES DF (ORIGINAL RULES FROM IMRAN) -----------------------------------------#    
