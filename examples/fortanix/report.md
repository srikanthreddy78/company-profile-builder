# Company profile report — Fortanix

- Run: `pb-20261002-5bsvmi`  ·  Status: **complete**  ·  Output: `/Users/srikanth/projects/company-profile-builder/.runs/pb-20261002-5bsvmi/company_brain.json`
- Website: https://www.fortanix.com/  ·  Product focus: Confidential Computing Platform
- Model calls: 16  ·  Tokens in/out: 179966/4385  ·  Estimated cost: $0.1857

## Coverage

| Section | Filled fields |
|---|---|
| company | 2/2 |
| product | 5/5 |
| customer | 7/7 |
| content_evidence | 4/4 |
| brand | 3/4 |

Grounding: 75/79 populated fields have verified evidence.
Ungrounded fields: `product.features_and_capabilities[3].name`, `product.features_and_capabilities[3].description`, `product.features_and_capabilities[3].how_it_works`, `product.features_and_capabilities[3].customer_benefit`

## Pages

- [fetched] https://www.fortanix.com/ (28985 chars)
- [fetched] https://www.fortanix.com/platform/confidential-computing (49642 chars)
- [fetched] https://www.fortanix.com/platform/confidential-computing-manager (19445 chars)
- [fetched] https://www.fortanix.com/platform (21363 chars)
- [fetched] https://www.fortanix.com/customers (19388 chars)
- [fetched] https://www.fortanix.com/customers/goldman-sachs-case-study (17911 chars)
- [fetched] https://www.fortanix.com/customers/beekeeperai-case-study (18383 chars)
- [fetched] https://www.fortanix.com/company/about (18306 chars)
- [fetched] https://www.fortanix.com/solutions/use-case/secure-data-driven-innovation (19226 chars)
- [fetched] https://www.fortanix.com/solutions/industry/data-security-for-banking-and-financial-services (20793 chars)

## Interview

1. **For the Confidential Computing Platform, who typically makes the buying decision?**  
   _Why:_ The website establishes industries and technical capabilities, but it does not clearly say which role usually owns purchase decisions for this product.  
   _Answer (answered):_ CISO, CIO, Head of Data Security and Compliance, VP of Cloud Infrastructure
2. **Who are the primary day-to-day users of the Confidential Computing Platform?**  
   _Why:_ The site describes platform management and attestation workflows, but it does not clearly separate the hands-on users from the economic buyers.  
   _Answer (answered):_ Security engineers, platform and DevOps engineers, cloud architects, and ML engineers running confidential AI workloads

## Remaining gaps

- `brand.terms_or_claims_to_avoid` — claims or terms the company does not want used

## Warnings

- `EVIDENCE_REJECTED` product.features_and_capabilities［3］: excerpt is not a verbatim quote from that page
- `EVIDENCE_REJECTED` customer.desired_outcomes: excerpt is not a verbatim quote from that page
