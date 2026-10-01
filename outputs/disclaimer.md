# Disclaimer

Exported by `scripts/export_outputs.py` from `mf_rag.prompts.DISCLAIMER`, which is the same constant the Streamlit UI renders (PRD FR-8.5). Copy the paragraph below verbatim into any other surface.

Facts-only. No investment advice. Information is taken from official public scheme pages and may change. Verify details on the AMC/AMFI site before acting.

## Refusal wording

Refusals are constants too (`mf_rag.prompts.REFUSAL_TEMPLATES`), so the exact words are reviewable in one file and identical on every run. Each takes a `{link}` placeholder, filled at runtime with an ingested URL:

- **advisory**: "I only share factual information from official scheme documents - I can't give investment advice or recommend a scheme. For general investor-education guidance, see: {link}."
- **portfolio**: "I only share factual information from official scheme documents - I can't give investment advice or recommend a scheme. For general investor-education guidance, see: {link}."
- **returns**: "I don't compute or compare returns. The official factsheet has the fund's reported performance: {link}."
- **out_of_corpus**: "I couldn't find that in the official HDFC documents I have access to, so I won't guess. You can check the official scheme page here: {link}."
- **pii**: "I can't accept personal identifiers such as PAN, Aadhaar, account numbers, OTPs, email addresses or phone numbers, and nothing you type is stored. Please rephrase your question without any personal details. For general investor-education guidance, see: {link}."

`REFUSAL_TEMPLATES['pii']` is `I can't accept personal identifiers such as PAN, Aadhaar, account numbers, OTPs, email addresses or phone numbers, and nothing you type is stored.` - the same constant the L1 PII guard returns verbatim.

Generated file - do not hand-edit; rerun `python scripts/export_outputs.py`.
