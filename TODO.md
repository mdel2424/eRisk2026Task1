# TODO

- [ ] Reorder graph to `extract_evidence -> update_beliefs -> assess_stop -> supervisor -> specialist`, look into passing information extracted forward to supervisor/specialist.
- [ ] Look into DAIC-WOZ for persona tuning (?), bolster personas.
- [ ] Tune lexical evidence cues for false negatives. Use eval examples for more cues.
- [ ] Rework update_beliefs/calibrator. Rethink component.
- [ ] Reduce extraction failures and tune early stopping strategy.
- [ ] Add BDI decisions traceability (log which decisions are made and why after each message).
- [ ] Work on getting more personas in evaluation (without slowing down too much?)
