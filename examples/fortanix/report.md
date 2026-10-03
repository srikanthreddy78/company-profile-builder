# Company profile report — Fortanix

- Run: `pb-20261002-lotlt2`  ·  Status: **complete**  ·  Output: `/Users/srikanth/projects/company-profile-builder/.runs/pb-20261002-lotlt2/company_brain.json`
- Website: https://www.fortanix.com/  ·  Product focus: Confidential Computing Platform
- Model calls: 16  ·  Tokens in/out: 189185/5739  ·  Estimated cost: $0.1882

## Coverage

| Section | Filled fields |
|---|---|
| company | 2/2 |
| product | 5/5 |
| customer | 7/7 |
| content_evidence | 4/4 |
| brand | 3/4 |

Grounding: 68/68 populated fields have verified evidence.

## Pages

- [fetched] https://www.fortanix.com/ (28985 chars)
- [fetched] https://www.fortanix.com/platform/confidential-computing (49642 chars)
- [fetched] https://www.fortanix.com/platform/confidential-computing-manager (19445 chars)
- [fetched] https://www.fortanix.com/platform (21363 chars)
- [fetched] https://www.fortanix.com/company/about (18306 chars)
- [fetched] https://www.fortanix.com/customers (19388 chars)
- [fetched] https://www.fortanix.com/customers/beekeeperai-case-study (18383 chars)
- [fetched] https://www.fortanix.com/customers/goldman-sachs-case-study (17911 chars)
- [fetched] https://www.fortanix.com/solutions/industry/data-security-for-healthcare (19975 chars)
- [fetched] https://www.fortanix.com/solutions/industry/data-security-for-banking-and-financial-services (20793 chars)

## Interview

1. **For the Confidential Computing Platform, who typically makes the buying decision on the customer side?**  
   _Why:_ The website shows regulated-industry use cases and technical platform details, but it does not clearly identify the buyer roles for this product.  
   _Answer (answered):_ CISO, CIO, Head of Data Security and Compliance, VP of Cloud Infrastructure
2. **Who are the primary day-to-day users of the Confidential Computing Platform after purchase?**  
   _Why:_ The website explains the platform and its industries, but it does not clearly separate hands-on users from the buyer roles.  
   _Answer (answered):_ Security engineers, platform and DevOps engineers, cloud architects, and ML engineers running confidential AI workloads

## Remaining gaps

- `brand.terms_or_claims_to_avoid` — claims or terms the company does not want used
- `product.features_and_capabilities[3].how_it_works` — how it works of capability 'Cross-cloud confidential computing control plane' is unknown
- `product.features_and_capabilities[4].how_it_works` — how it works of capability 'Confidential AI protection' is unknown

## Warnings

- `EVIDENCE_REJECTED` product.features_and_capabilities［3］.name: excerpt does not mention the field value; quote the passage the value comes from
