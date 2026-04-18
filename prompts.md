# Prompt Sensitivity Analysis

To check whether pseudo-label quality is sensitive to prompt design, we evaluated three prompt variants with GPT-4o on the dev split of all 10 events. We used the dev split rather than the test split because prompt selection is a model-selection decision, and test data should be reserved for final reporting.

**RULES_1** uses compact, single-line class definitions (for example, "caution_and_advice: warnings/instructions/tips").
**RULES_2** adds a framing instruction ("Pick ONE label for the tweet's PRIMARY INTENT") with slightly expanded definitions.
**RULES_3** provides detailed multi-line guidance per class with explicit Definition, Include, and Exclude criteria (about 3–4× longer than RULES_1).

Table 4 reports the per-event results. RULES_1 achieves the highest averaged Macro-F1 (0.613), narrowly ahead of RULES_2 (0.612), while RULES_3 lags at 0.601. The mean difference between RULES_1 and RULES_2 is 0.001, within run-to-run noise, and the largest per-event gap across the three variants is 0.055 (Hurricane Florence 2018).

These results indicate that GPT-4o is reasonably robust to prompt phrasing for this task, and that more detailed inclusion and exclusion criteria do not help. RULES_1 was chosen for pseudo-label generation because it is the simplest and cheapest variant, and the dev-set comparison confirms it is among the best-performing variants.

---

## Table 4: Prompt Sensitivity (Macro-F1 on Dev Split)

| Event                     |   RULES_1 |   RULES_2 |   RULES_3 |
| ------------------------- | --------: | --------: | --------: |
| California Wildfires 2018 | **0.608** |     0.589 |     0.585 |
| Canada Wildfires 2016     |     0.581 | **0.588** |     0.580 |
| Cyclone Idai 2019         | **0.557** |     0.552 |     0.519 |
| Hurricane Dorian 2019     |     0.581 | **0.608** |     0.563 |
| Hurricane Florence 2018   |     0.706 | **0.711** |     0.656 |
| Hurricane Harvey 2017     |     0.617 | **0.619** |     0.615 |
| Hurricane Irma 2017       |     0.589 |     0.587 | **0.592** |
| Hurricane Maria 2017      |     0.621 | **0.629** |     0.622 |
| Kaikoura Earthquake 2016  | **0.668** |     0.655 |     0.649 |
| Kerala Floods 2018        |     0.602 |     0.584 | **0.630** |
| **Mean**                  | **0.613** |     0.612 |     0.601 |

**Table 4.** Prompt sensitivity: GPT-4o Macro-F1 on the dev split under three prompt variants.

* **RULES_1:** compact single-line definitions
* **RULES_2:** medium-detail with "PRIMARY INTENT" framing
* **RULES_3:** detailed Definition/Include/Exclude guidance per class
* Best result per row is shown in **bold**

---

## Supporting Data (Reference Only)

**Source:** `rules/humaid_rules.py`

### RULES_1

* caution_and_advice: warnings/instructions/tips
* displaced_people_and_evacuations: evacuations, relocation, shelters
* infrastructure_and_utility_damage: damage/outages to roads/bridges/power/water/buildings
* injured_or_dead_people: injuries, casualties, fatalities
* missing_or_found_people: missing or found persons
* requests_or_urgent_needs: asking for help/supplies/SOS
* rescue_volunteering_or_donation_effort: offering help, donation, organizing aid
* sympathy_and_support: prayers/condolences, no actionable info
* other_relevant_information: on-topic but none of the above
* not_humanitarian: unrelated to disasters/aid

**Tie-break:** prefer actionable class when in doubt.

---

### RULES_2

**Instruction:** Pick ONE label for the tweet's PRIMARY INTENT.

* caution_and_advice: warnings/instructions/tips about the disaster
* displaced_people_and_evacuations: evacuation/relocation/shelter/displaced
* infrastructure_and_utility_damage: damage/outages to roads/buildings/power/water/comms caused by the disaster
* injured_or_dead_people: injuries/casualties/deaths
* missing_or_found_people: explicit missing OR found/reunited persons
* requests_or_urgent_needs: asking for help/supplies/services (need/urgent/sos)
* rescue_volunteering_or_donation_effort: offering help; organizing rescues/donations/volunteers/events
* sympathy_and_support: prayers/condolences/morale support (no logistics)
* other_relevant_information: on-topic facts/stats/official updates when none above fits
* not_humanitarian: unrelated to disasters or unclear context

---

### RULES_3

(Condensed for readability; full version in source file)

* **caution_and_advice:** warnings with actionable instructions
* **displaced_people_and_evacuations:** evacuation/shelter-related content
* **infrastructure_and_utility_damage:** physical damage/outages
* **injured_or_dead_people:** casualties/fatalities
* **missing_or_found_people:** missing/found individuals
* **requests_or_urgent_needs:** urgent help/supplies requests
* **rescue_volunteering_or_donation_effort:** organizing/offering help
* **sympathy_and_support:** prayers/condolences
* **other_relevant_information:** general disaster-related info
* **not_humanitarian:** unrelated content
