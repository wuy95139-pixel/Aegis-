"""
Sample data fixtures
====================
Reusable sample documents and data for demos and tests.
"""

from pathlib import Path


def create_sample_meeting_doc(output_dir: str = "./output") -> str:
    """Create a sample meeting notes file for demo/testing purposes.

    Returns the file path to the created document.
    """
    content = """# Product R&D Weekly Meeting Notes
Date: 2024-06-03 (Monday) 14:00-15:30
Attendees: Alice (PM), Bob (Frontend), Charlie (Backend), Diana (Designer), Eve (QA)

## 1. Last Week Review
- Alice reported Q2 product roadmap progress at 85%
- Bob completed frontend refactoring of the user dashboard, 30% perf improvement
- Charlie completed API v2.0 development and unit tests
- Diana delivered the first draft of the new brand visual design

## 2. This Week's Priorities
- Bob needs to complete responsive mobile layout by June 8
- Charlie needs to coordinate with Bob on API v2.0 integration, deadline June 10
- Diana needs to deliver the final brand visual design by June 5
- Eve needs to complete regression testing of new features by June 12

## 3. Issues & Risks
- Users report slow page loading, needs joint frontend/backend investigation
- Color discrepancy between design mockups and implementation needs standardization

## 4. Decisions
- Code Review sprint starting next Monday (June 10), led by Charlie
- Adopt new UI component library, Diana to produce design spec doc by June 15
- Eve to set up automated testing pipeline, target end of June

## 5. Next Meeting
- Time: June 11 (Tuesday) 14:00
- Agenda: Code Review kickoff + Q2 sprint planning
"""
    filepath = Path(output_dir) / "sample_meeting_notes.md"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return str(filepath)
