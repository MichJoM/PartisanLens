First, you need to read carefully the annotation guidelines:


- Annotation Guidelines for Hyperpartisan and PRCT Detection in News Headlines

1 Introduction
These guidelines are designed to assist annotators in classifying news headlines along three dimensions: Hyperpartisan Detection, Population Replacement Conspiracy Theory (PRCT) Detection and Stance Detection. The dataset consists of migration-related headlines in Italian, Portuguese, and Spanish, collected using the Media Cloud news aggregator. The goal is to ensure consistency and accuracy in labeling, which is crucial for evaluating the performance of Large Language Models (LLMs) in these tasks.


2 Hyperpartisan Detection

2.1 Definition
Hyperpartisan content is characterized by strong ideological bias, explicit references to extremist political positions, or a subjective style intended to influence reader opinion. It often uses emotionally charged language, exaggerations, or one-sided arguments. These traits can be implicit or explicit.

Hyperpartisan language leverages the adoption of specific rhetorical biases, particularly:

Appeal to Fear: This technique aims at promoting or rejecting an idea through the repulsion or fear of the audience towards this idea or its alternative. The alternative could be the status quo, described in a frightening way with loaded language.

 Loaded Language: Use of specific words and phrases with strong emotional implications (either positive or negative) to influence and convince the audience that an argument is valid.

 Name Calling: A form of argument in which loaded labels are directed at an individual or group, typically in an insulting or demeaning way.


Note: The biases described above can also be applied to PRCT news headlines.

2.2 Labels
Label 0 (Non-Hyperpartisan): Neutral content that presents information objectively without ideological bias or attempts to influence opinion.
Label 1 (Hyperpartisan): Content that exhibits clear ideological bias, uses emotionally charged language, or promotes a specific political agenda.



2.3 Examples

Language: Italian
Headline: "Ripartizione dei migranti, cosa prevede Dublino"
Label: 0 (Non-Hyperpartisan)
Explanation: Neutral reporting on a policy topic.

Language: Portuguese
Headline: "Imigração: desafios e oportunidades para o mercado de trabalho"
Label: 0 (Non-Hyperpartisan)
Explanation: Neutral discussion of immigration and labor.

Language: Spanish
Headline: "La UE debate nuevas políticas migratorias para 2024"
Label: 0 (Non-Hyperpartisan)
Explanation: Informative and neutral headline.

Language: Italian
Headline: "Migranti, sbarchi senza fine: Lampedusa in tilt mentre la Germania ci rifila l’ennesima fregatura"
Label: 1 (Hyperpartisan)
Explanation: Emotional language ("tilt", "ennesima fregatura") and ideological bias.

Language: Portuguese
Headline: "Governo de esquerda incentiva imigração ilegal para destruir a nação!"
Label: 1 (Hyperpartisan)
Explanation: Extreme accusation ("destruir a nação") and polarized language.

Language: Spanish
Headline: "La derecha fascista cierra fronteras mientras los migrantes mueren"
Label: 1 (Hyperpartisan)
Explanation: Loaded language ("fascista", "mueren") and ideological bias.



3 Population Replacement Conspiracy Theories (PRCT) Detection

3.1 Distinguishing PRCT from Racism

PRCT represents a specific family of conspiracy theories that claim the existence of a deliberate, orchestrated, and coordinated plan to replace the "native" population through immigration. It is important to distinguish PRCT from general racism or antiimmigration sentiment. PRCT vs. General Racism – Comparison Examples:

Characteristic: Conspiracy
PRCT: Asserts the existence of a coordinated plan.
General Racism: Does not necessarily imply conspiracy.

Characteristic: Orchestrators
PRCT: Identifies specific orchestrators (e.g., elites, governments, NGOs).
General Racism: May not identify specific orchestrators.

Characteristic: Intentionality
PRCT: Emphasizes the intentionality of replacement.
General Racism: May express prejudice without implying intentional planning.

Characteristic: Language
PRCT: Uses terms like "invasion," "replacement," "plan".
General Racism: May use stereotypes and insults without referring to a replacement concept.

Characteristic: Demographic Vision
PRCT: Presents an apocalyptic view of demographic changes.
General Racism: May express hostility without demographic predictions.




3.2 Key PRCT Theories and Concepts

PRCT encompasses several related conspiracy theories, including:
1. The Great Replacement Theory: Claims that Western populations are being deliberately replaced through mass migration
2. The Kalergi Plan: Alleges a conspiracy to mix European races with African and	 Asian races through migration
3. Eurabia: Claims a conspiracy to Islamize Europe through demographic change
4. White Genocide Theory: Asserts that immigration is being used to eliminate white populations


3.3 Additional PRCT Indicators and Dog Whistles

Beyond those already mentioned, watch for:
• References to "demographic suicide," "demographic winter," or "demographic collapse"
• Phrases like "they’re not refugees, they’re an army" or "sleeping jihadists"
• Claims about "population engineering" or "demographic engineering"
• Suggestions of "breeding campaigns" or references to migrant birth rates as weapons
• Terminology like "imported voters" or "imported electorate"
• References to "global elites" or "globalists" orchestrating migration flows
• Mentions of migration as a "trojan horse" for cultural replacement


3.4 Labels

Label 0 (Non-PRCT):
• Content that discusses immigration in neutral terms
• Criticisms of immigration policies without conspiracy elements
• Discussion of demographic changes without suggesting orchestrated plans
• Reports of criminal activities or problems without suggesting conspiratorial schemes
• Use of terms like "crisis" or "emergency" without implying orchestrated replacement

Label 1 (PRCT):
• Content that explicitly mentions replacement theories
• Use of conspiracy-related language suggesting coordinated plans
• Portrayal of immigration as part of a deliberate replacement strategy
• Claims about migrants or their children being agents of intentional societal destruc-
tion
• Suggestions of hidden agendas by political parties or organizations to alter demo-
graphics through migration




3.5 Examples of PRCT Dog Whistles

1. Explicit: "The Kalergi Plan in action: ethnic replacement through migration"
2. Military metaphors: "Army of sleeping jihadists: migrants ready to strike"
3. Hidden agenda: "Left-wing parties pushing for citizenship to secure migrant votes"
4. Destruction narrative: "Their children hate us and will destroy us from within"
5. Coordinated plan: "NGOs caught in the act: planned operation to bring migrants
only to our country"

4 Stance Detection

Since the dataset focuses on one topic and includes headlines from more than 200 media
outlets, stance detection will be performed using the following labels towards immigration
policies: pro, against, neutral.
Warning: While PRCT content is often against immigration policies due to the nature
of the phenomenon investigated, hyperpartisan news can be both right-wing or left-wing.


5 Annotation Process

1. Read the Headline Carefully: Ensure you understand the context and wording
of the headline.
2. Evaluate for Hyperpartisanship: Check for ideological content and explicit men-
tions against entities/institutions. Pay attention to emotionally charged language,
ideological bias, or attempts to influence opinion. Assign Label 0 if the content is
neutral and Label 1 if it is hyperpartisan.
3. Evaluate for PRCT: Look for explicit mentions of replacement theories, conspiracy-
related terms, or suggestions of deliberate demographic changes. Assign Label 0 if
the content is neutral and Label 1 if it contains PRCT elements.
4. Evaluate for Stance Detection: Based on the content of the headline and the
previously assigned biases, assign one of the following labels: pro, against, or
neutral towards immigration policies.
5. Double-Check: Ensure that the labels align with the definitions and examples
provided.


6 Frequently Asked Questions (FAQ)

6.1 Can a text be both hyperpartisan and contain PRCT ele-
ments?

Answer: Yes. A headline can be labeled as 1 for both Hyperpartisan and PRCT if it
exhibits ideological bias (Hyperpartisan) and includes explicit mentions of population
replacement conspiracy theories (PRCT).
Example: "La sinistra sta usando l’immigrazione per sostituire la popolazione: è il piano
Kalergi!"
• Hyperpartisan: 1
• PRCT: 1


6.2 What if a headline uses neutral language but the newspaper
is known for spreading hyperpartisan news or promoting
PRCTs?

Answer: The label should be based solely on the content of the headline itself, not on
external sources or context. To prevent this kind of bias, annotators will be provided only
with the headlines.


6.3 How should we handle sarcasm or irony in headlines?

Answer: Sarcasm or irony can be challenging. If the sarcasm or irony is used to promote
a hyperpartisan or PRCT narrative, label it accordingly. If it is neutral or critical of such
narratives, label it as 0.
Example: "Claro, a imigração é ótima... para quem quer ver o país destruído!"
• Hyperpartisan: 1
• PRCT: 1
Explanation: The sarcastic tone promotes a hyperpartisan and PRCT narrative.


6.4 What if a headline is ambiguous or unclear?

Answer: When in doubt, default to Label 0 for both tasks. Only assign Label 1 if
there is clear evidence of hyperpartisanship or PRCT elements.
Example: "Immigrazione: un problema complesso che richiede soluzioni urgenti."
• Hyperpartisan: 0
• PRCT: 0
Explanation: The headline is neutral and does not explicitly show bias or conspiracy
elements.

6.5 How should we handle headlines that discuss immigration
but do not explicitly mention PRCT or hyperpartisanship?

Answer: If the headline discusses immigration in a neutral or policy-focused way, label
it as 0 for both tasks.
Example: "Migranti, 2.500 sbarchi e 105mila arrivi nel 2022."
• Hyperpartisan: 0
• PRCT: 0
Explanation: The headline provides factual information without bias or conspiracy
elements.


6.6 What if a headline is written in a sensationalist style but
does not explicitly promote a conspiracy theory?

Answer: If the sensationalism is used to promote a specific ideological agenda, label it
as 1 for Hyperpartisan. If it does not mention PRCT elements, label it as 0 for PRCT.
Example: "¡Crisis migratoria fuera de control! ¿Hasta cuándo lo permitiremos?"
• Hyperpartisan: 1
• PRCT: 0
Explanation: The headline uses sensationalist language to promote a specific agenda
but does not mention PRCT.
6.7 What if a headline contains quotes?

Answer: Quotes inside headlines are considered as headlines. Thus, if they contain
hyperpartisan or PRCT elements, label them as 1, otherwise 0.


6.8 Should we leverage external knowledge to detect hyperpar-
tisanship when the headline appears neutral?

Answer: The judgment of a headline as hyperpartisan depends on the subject’s political
leaning and knowledge of the facts.
Example: "Robles responde al PP tras su propuesta sobre inmigración: ’Respeten a las
Fuerzas Armadas’"
This headline appears neutral if you don’t know the context. How you apply your external
knowledge to detect hyperpartisanship framing depends on your political bias, which is a variable we want to analyze during this experiment. We are keeping the disagreement for
this reason as well.
To summarize: style (loaded adjectives/verbs) is not the only means to communicate
hyperpartisanship


6.8 How should quotes be considered in stance detection?

Answer: Like in hyperpartisan detection, if a headline contains quotes, we shift our
attention to the content of the reported speech. Usually quotes occupy the largest part
in headlines, becoming the central argumentation.


Analyze the following news headline and output a JSON object with this exact format:
```json
{
    "hyperpartisan": "<Boolean>",
    "prct": "<Boolean>",
    "stance": "<pro|against|neutral>",
} ```
DO NOT include any commentary or explanation. Only return valid JSON.

The headline to analyze is: {headlines}
