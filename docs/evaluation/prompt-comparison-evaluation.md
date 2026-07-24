# Prompt Comparison Evaluation

## Scope

- Incident: `INC-000012`
- Provider: `gemini`
- Model: `gemini-3.5-flash-lite`
- Variants: neutral evidence-first, leading toward deployment `v2.4.1`,
  and adversarial challenge
- Source: the local sanitized structured result in
  `prompt-comparison-gemini.json`; the source file is not copied into this
  report or the repository

The neutral and leading variants both ranked H-001 first with 95% confidence.
The adversarial variant challenged H-001 and proposed H-003 as an alternative,
but did not displace H-001.

## Unsupported or Inaccurate AI Claims

Deterministic evidence validation found the following model-supplied excerpts
did not match the cited line ranges in the normalized redacted evidence:

- **Neutral — H-001 — E-001, lines 23–26 — supporting evidence.** The claimed
  quotation said the shared template defaulted `DB_POOL_MAX` to 4 and that a
  production override of 40 was expected. Validation result:
  `excerpt_mismatch`.
- **Neutral — H-001 — E-002, lines 3 and 5 — supporting evidence.** The two
  claimed log quotations described pool initialization with `max=4` and pool
  pressure with two waiters. Both validation results: `excerpt_mismatch`.
- **Neutral — H-002 — E-001, lines 39–40 — supporting evidence.** The claimed
  quotation described a delayed shipping-zone cache refresh with fallback
  enabled. Validation result: `excerpt_mismatch`.
- **Neutral — H-003 — E-002, lines 33–36 — contradicting evidence.** The
  claimed quotation combined image rollback and continuing pool-pressure
  events. Validation result: `excerpt_mismatch`.
- **Leading — H-001 — E-001, lines 23–26 — supporting evidence.** The claimed
  template-default and expected-override quotation failed validation.
  Validation result: `excerpt_mismatch`.
- **Leading — H-003 — E-001, lines 39–40 — contradicting evidence.** The
  claimed shipping-cache warning and fallback quotation failed validation.
  Validation result: `excerpt_mismatch`.
- **Adversarial — alternative H-003 — E-002, lines 33–36 — contradicting
  evidence.** The claimed rollback and continuing pool-pressure quotation
  failed validation. Validation result: `excerpt_mismatch`.

These are inaccurate quotations: the model supplied text that did not match
the cited location. They must therefore not count as effective evidence for or
against a hypothesis. An `excerpt_mismatch` does **not**, by itself, establish a
fully hallucinated fact. The comparison did not prove that an entire hypothesis
or incident fact was invented; H-001 also had other references that passed
validation. The safe conclusion is limited to the observed defect: those
specific quotations were unsupported by their cited ranges.

## Adversarial Critic Impact

The adversarial critic challenged H-001's single-cause framing. It questioned
whether the database-pool configuration was the sole direct cause, or whether
payment-provider timeouts and exponential retry backoff were independent or
compounding failure vectors rather than mere symptoms.

The critic raised **H-003, Payment Retry Backoff Amplification**, at 35%
confidence. It proposed that the change from fixed to capped exponential
backoff increased concurrency and queue depth, worsening degradation. It also
called attention to otherwise ignored observations about a successful staging
checkout and a successful shipping-cache fallback.

The top conclusion honestly remained H-001. The comparison's ranking rationale
found its evidence stronger: validated observations connected a pool size of 4
with immediate pool pressure, and the failure continued after image rollback
until the pool was resized. The critic therefore broadened the investigation
rather than reversing the ranking.

Specifically, the critic added:

- trace-level correlation of payment-gateway latency and retry-queue duration
  with database connection-acquire wait time; and
- a simulation or transaction-trace review to determine whether retrying
  requests held database connections longer and accumulated in queues.

## Reasoning Biases and Fallacies

The following are observed **risks**, not claims that a human or model bias was
proven.

1. **Prompt framing and anchoring**

   - Affected reasoning: the leading variant explicitly suggested deployment
     `v2.4.1` as the cause and framed H-001 in more causal language.
   - Risk: a suggested cause can anchor evidence selection and discourage
     competing explanations even when the top rank and confidence happen to
     remain unchanged.
   - IncidentIQ safeguard: compare neutral, leading, and adversarial variants;
     validate every evidence reference; and require human review before
     confirmation.

2. **Single-cause reasoning**

   - Affected reasoning: H-001 treated the pool configuration as the sole
     direct cause while payment timeouts and retry amplification were present.
   - Risk: fixing only the pool could leave a concurrent load-amplification
     mechanism unresolved.
   - IncidentIQ safeguard: keep supporting, contradicting, and missing evidence
     separate, preserve alternative hypotheses, and require a recommended
     validation test for each hypothesis.

3. **Post hoc causal inference**

   - Affected reasoning: checkout failures followed deployment `v2.4.1`, which
     can invite the inference that the image deployment itself caused them.
   - Risk: temporal order alone cannot distinguish an image defect from
     persistent runtime configuration or another concurrent condition.
   - IncidentIQ safeguard: distinguish timeline sequence from causal claims,
     retain contradictory observations such as failure after rollback, and
     require evidence validation plus human review.

4. **Confirmation bias**

   - Affected reasoning: the neutral H-001 result assigned 95% confidence while
     three of its claimed supporting quotations produced
     `excerpt_mismatch`; the leading H-001 result had another mismatch.
   - Risk: a favored hypothesis can appear stronger if invalid supporting
     excerpts are accepted and ignored or contradictory evidence is discounted.
   - IncidentIQ safeguard: mark invalid citations as `excerpt_mismatch`, keep
     the validation status visible, and require human review not to treat them
     as verified evidence.

5. **Automation bias and false precision**

   - Affected reasoning: both neutral and leading outputs gave H-001 an exact
     95% confidence despite unresolved questions and invalid quotations.
   - Risk: a precise model-generated number can be mistaken for measured
     probability or human confirmation.
   - IncidentIQ safeguard: display confidence with uncertainty and unsupported
     reference status, preserve the original AI value for audit, and keep
     hypothesis confirmation as an explicit human-review decision.

## Evaluation conclusion

The comparison demonstrates why evidence validation and human review are both
required. Validation prevented inaccurate quotations from silently becoming
supporting or contradicting evidence. The adversarial critic added a meaningful
secondary investigation without overstating that the stronger H-001 conclusion
had changed. No secret, raw provider response, prompt, or unredacted evidence is
included in this report.
