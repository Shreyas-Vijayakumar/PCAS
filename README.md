## Known Limitations & Assumptions

### P1 — Callout Generator
- **Runway selection**: Wind-based selection uses METAR circular mean.
  On 2020-10-22, winds were light (~4 kts) at ~82° crosswind to both runways.
  Algorithm selects Runway 08 (marginally better headwind component: +0.56 kts).
  Real pilots may use either runway under calm conditions.
  Production system should incorporate AWOS/ATIS broadcast or pilot-reported runway.
  
- **Phase classification**: Pattern leg detection uses heading tolerance ±35°.
  GA aircraft in turns may briefly misclassify between legs.
  
- **Timezone**: Both ADS-B and audio confirmed local EDT per TartanAviation paper.
  No UTC correction applied.