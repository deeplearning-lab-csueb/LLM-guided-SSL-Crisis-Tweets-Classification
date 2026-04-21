# rules/humaid_rules.py

"""
HumAID zero-shot rules (project-local, experiment-focused).

- Put your evolving rule variants here: RULES_5 ... RULES_10
- Keep them short to control token costs.
- The label order matches README and humaidclf.prompts.LABELS.
"""

# Baseline template we can always fall back to
RULES_BASELINE = """
- caution_and_advice: Reports of warnings issued or lifted, guidance and tips related to the disaster.
- displaced_people_and_evacuations: People who have relocated due to the crisis, even for a short time...
- infrastructure_and_utility_damage: Reports of any type of damage to infrastructure such as buildings, houses,...
- injured_or_dead_people: Reports of injured or dead people due to the disaster.
- missing_or_found_people: Reports of missing or found people due to the disaster.
- requests_or_urgent_needs: Reports of urgent needs or supplies such as food, water, clothing, money,...
- rescue_volunteering_or_donation_effort: Reports of any type of rescue, volunteering, or donation efforts...
- sympathy_and_support: Tweets with prayers, thoughts, and emotional support.
- other_relevant_information: on-topic but none of the above
- not_humanitarian: If the tweet does not convey humanitarian aid-related information.
""".strip()

# === Experiment variants ===

RULES_1 = """
- caution_and_advice: warnings/instructions/tips
- displaced_people_and_evacuations: evacuations, relocation, shelters
- infrastructure_and_utility_damage: damage/outages to roads/bridges/power/water/buildings
- injured_or_dead_people: injuries, casualties, fatalities
- missing_or_found_people: missing or found persons
- requests_or_urgent_needs: asking for help/supplies/SOS
- rescue_volunteering_or_donation_effort: offering help, donation, organizing aid
- sympathy_and_support: prayers/condolences, no actionable info
- other_relevant_information: on-topic but none of the above
- not_humanitarian: unrelated to disasters/aid
Tie-break: prefer actionable class when in doubt.
""".strip()

RULES_2 = """
Pick ONE label for the tweet's PRIMARY INTENT.

- caution_and_advice: warnings/instructions/tips about the disaster
- displaced_people_and_evacuations: evacuation/relocation/shelter/displaced
- infrastructure_and_utility_damage: damage/outages to roads/buildings/power/water/comms caused by the disaster
- injured_or_dead_people: injuries/casualties/deaths
- missing_or_found_people: explicit missing OR found/reunited persons
- requests_or_urgent_needs: asking for help/supplies/services (need/urgent/sos)
- rescue_volunteering_or_donation_effort: offering help; organizing rescues/donations/volunteers/events
- sympathy_and_support: prayers/condolences/morale support (no logistics)
- other_relevant_information: on-topic facts/stats/official updates when none above fits
- not_humanitarian: unrelated to disasters or unclear context
""".strip()

RULES_3 = """
- caution_and_advice
  Definition: Action-oriented warnings, instructions, or tips telling people what to do/not do during the disaster.
  Include: Imperatives (“evacuate”, “avoid…”, “do not…”, “boil water”, “seek shelter”); agency alerts with directives.
  Exclude: Pure situation updates during the disaster without directives → other_relevant_information.

- displaced_people_and_evacuations
  Definition: People being evacuated/relocated, shelters open, or explicit evacuation orders during the disaster.
  Include: “Evacuation order issued…”, “Shelter at…”, “Families relocated to…”.
  Exclude: General travel closures or roadblocks without people movement during the disaster → infrastructure_and_utility_damage.

- infrastructure_and_utility_damage
  Definition: Physical damage or outages to roads/bridges/buildings/power/water/communications during the disaster.
  Include: “Bridge collapsed…”, “Power is out…”, “Water main burst…”.
  Exclude: Human impact (injuries/deaths) during the disaster → injured_or_dead_people; general updates without damage during the disaster → other_relevant_information.

- injured_or_dead_people
  Definition: Mentions of injuries, casualties, fatalities, body counts during the disaster.
  Include: “3 injured…”, “2 fatalities reported…”.
  Exclude: Sympathy without explicit injury/death info during the disaster → sympathy_and_support.

- missing_or_found_people
  Definition: People reported missing or found/located during the disaster.
  Include: “Missing person…”, “Have you seen…”, “Found safe”.
  Exclude: Volunteer search coordination without a missing person mention during the disaster → rescue_volunteering_or_donation_effort.

- requests_or_urgent_needs
  Definition: Asking for life-sustaining help/supplies/SOS for affected people during the disaster.
  Include: “We need water/food/medicine”, “Please send help”, “Trapped, need rescue”.
  Exclude: Calls for volunteers/donations to help others during the disaster → rescue_volunteering_or_donation_effort.

- rescue_volunteering_or_donation_effort
  Definition: Offering help, organizing aid, collecting donations, recruiting volunteers during the disaster.
  Include: “Volunteers needed”, “Accepting donations at…”, “Team deploying supplies”.
  Exclude: Personal survival needs (“we need food/water now”) during the disaster → requests_or_urgent_needs.

- sympathy_and_support
  Definition: Condolences, prayers, encouragement without actionable info during the disaster.
  Include: “Praying for…”, “Stay strong…”.
  Exclude: Any concrete info about damage, injuries, evacuations, or advice during the disaster → other_relevant_information.

- other_relevant_information
  Definition: On-topic disaster/humanitarian context that is informative but not covered above.
  Include: Event confirmation, location/time/scale updates, hazard observations, maps w/o directives.
  Exclude: If any action/impact criterion above is met, prefer that specific label.

- not_humanitarian
  Definition: Unrelated to disasters/aid or purely metaphorical/entertainment use of disaster terms.
  Include: Jokes, metaphors, historical trivia, movie/game talk.
  Exclude: If it references a real ongoing diaster, prefer other_relevant_information over this. 
"""

RULES_4 = """
- caution_and_advice: warnings/instructions/tips
- displaced_people_and_evacuations: evacuations, relocation, shelters
- infrastructure_and_utility_damage: damage/outages to roads/bridges/power/water/buildings
- injured_or_dead_people: injuries, casualties, fatalities
- missing_or_found_people: missing or found persons
- requests_or_urgent_needs: asking for help/supplies/SOS
- rescue_volunteering_or_donation_effort: offering help, donation, organizing aid
- sympathy_and_support: prayers/condolences, no actionable info
- other_relevant_information: on-topic but none of the above
- not_humanitarian: unrelated to disasters/aid
Tie-break: prefer actionable class when in doubt.
""".strip()

# Optional: a tiny registry so you can fetch by name
RULES_REGISTRY = {
    "BASELINE": RULES_BASELINE,
    "RULES_1": RULES_1,
    "RULES_2": RULES_2,
    "RULES_3": RULES_3,
    "RULES_4": RULES_4,
}

def get_rule(name: str) -> str:
    """Return the rule text by key in RULES_REGISTRY."""
    try:
        return RULES_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown rule name: {name}. Available: {list(RULES_REGISTRY)}") from e
