# Data Folder

This folder contains anonymized data from a two-phase controlled experiment investigating whether AI-assisted development affects the maintainability of code. The full methodology, experimental design, and analysis procedures are documented in the accompanying papers.

## Files

---

### `task1_preprocessed.csv` — Phase 1: Feature Implementation

This file contains data from Task 1, in which participants implemented a feature in a substandard Java/Spring Boot application with or without the assistance of an AI coding tool.

**Columns:**

- `github`: Anonymized GitHub ID
- `successors`: Number of Task 2 participants who extended this submission
- `unresponsives`: Whether the participant signed up but never started
- `incompletes`: Whether the participant started but didn’t submit a valid solution
- `ai`: Boolean indicating use of an AI assistant (`True`/`False`)
- `codehealth`: CodeScene CodeHealth score (1–10)
- `coverage`: Final statement test coverage (0–1)
- `test_runs`: Number of acceptance test executions
- `time`: Measured task duration (in minutes)
- `estimated_time`: Self-reported time spent on the task
- `entry-demo-1` to `entry-demo-5`: Prescreening demographic responses:
  - Gender
  - Age group
  - Country
  - Role (e.g., developer, student)
  - Java proficiency
- `entry-ai-1` to `entry-ai-10`: Likert-scale responses on AI experience and preference
- `entry-date`: Timestamp when the participant started Task 1
- `exit-uninterrupted`: Whether the task was completed in one sitting
- `exit-used-ai`: Participant-reported AI usage during the task
- `exit-ai-text`: AI tool used (if any)
- `exit-ai-frequency`: Frequency of interaction with the AI assistant
- `exit-resemblance`: Perceived realism of the task
- `exit-tools-text`: Free-text input on additional development tools used
- `exit-space-1` to `exit-space-10`: Likert-scale responses aligned with the SPACE productivity framework
- `exit-final-text`: Optional free-text comment at the end of the task
- `exit-date`: Task submission timestamp
- `email-text`: Optional information shared via follow-up email communication (e.g., clarifications, time estimates)
- `gh-commits`, `gh-additions`, `gh-deletions`, `gh-changes`, `gh-changed-files`: Git activity metrics
- `gh-adds-unit-test`, `gh-adds-behavioural-test`, `gh-adds-functional`, `gh-adds-sql`, `gh-adds-logging`, `gh-adds-exception-handling`, `gh-adds-dependency`: Boolean flags for types of code additions
- `clean_time`: preprocessed time used for the analysis
- `pp_mran`: mean value of the exit-space-1 to exit-space-10, inverting items 4 and 4

---

### `task2_preprocessed.csv` — Phase 2: Code Evolution (Randomized Controlled Trial)

This file contains data from Task 2, in which new participants extended existing codebases originally written in Task 1. The main independent variable is whether the base code was created with the help of an AI assistant.

**Columns:**

- `github`: Anonymized GitHub ID
- `predecessor`: GitHub ID of the Task 1 submission that was extended
- `treatment`: Boolean (`True` = code evolved was AI-assisted)
- `codehealth`, `codehealth_diff`: Final CodeScene score and its delta from the predecessor
- `coverage`, `coverage_diff`: Final test coverage and delta
- `test_runs`: Number of acceptance test executions
- `time`, `estimated_time`: Measured and self-estimated task duration (in minutes)
- `entry-demo-1` to `entry-demo-5`: Prescreening demographics (as in Task 1)
- `entry-ai-1` to `entry-ai-10`: AI experience and preference ratings (used for covariate analysis)
- `entry-date`: Task 2 start timestamp
- `exit-uninterrupted`: Whether the task was completed in one session
- `exit-used-ai`: Should be `False` (Task 2 disallowed AI use)
- `exit-ai-text`, `exit-ai-frequency`: Should be empty or "None"
- `exit-resemblance`: Perceived realism of the task
- `exit-tools-text`: Free-text input on any additional tools used
- `exit-space-1` to `exit-space-10`: Likert-scale responses aligned with the SPACE framework
- `exit-final-text`: Optional comment at the end of the task
- `exit-date`: Submission timestamp
- `email-text`: Optional information shared via follow-up email (e.g., explanations, timing clarifications)
- `gh-commits`, `gh-additions`, `gh-deletions`, `gh-changes`, `gh-changed-files`: Git activity metrics
- `gh-adds-unit-test`, `gh-adds-behavioural-test`, `gh-adds-functional`, `gh-adds-sql`, `gh-adds-logging`, `gh-adds-exception-handling`, `gh-adds-dependency`: Flags for types of code additions
- `clean_time`: preprocessed time used for the analysis
- `pp_mran`: mean value of the exit-space-1 to exit-space-10, inverting items 4 and 4

---

### `entry.csv` - Prescreening questionnaire

This file contains prescreening information. The columns are described above.

### `task1_raw.csv` - Prescreening questionnaire

Added for completeness.

### `task2_raw.csv` - Prescreening questionnaire

Added for completeness.

## Notes

- All participant IDs (`github`) have been anonymized.
- Columns prefixed with `entry-` are from the prescreening questionnaire.
- Columns prefixed with `exit-` are from the exit survey after each task.
- `email-text` fields are **not** email addresses; they contain optional clarifications provided during post-task follow-up.
- Free-text responses (`exit-tools-text`, `exit-final-text`, `email-text`) may contain feedback and should be treated cautiously in downstream analysis.

For full methodological details and instrumentation, see the registered report and the intermediate summary in the `paper/` folder.
