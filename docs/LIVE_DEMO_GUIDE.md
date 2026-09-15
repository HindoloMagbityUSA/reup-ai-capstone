# ReUP AI live-demo guide

## Before presentation day

1. Open the entire `ReUP_AI_Demo_Package 2` folder in VS Code.
2. Double-click `START_REUP_DEMO.command` while connected to the internet. The first run creates `.venv` and installs the required packages.
3. Confirm that the browser opens the ReUP AI Listing Assistant.
4. Upload the three recommended examples below and verify that the expected field is suggested.

## Recommended demonstration sequence

1. Upload `demo_images/concrete.jpg` to show a material-type suggestion.
2. Upload `demo_images/component_door.jpg` to show a door component suggestion.
3. Upload `demo_images/condition_visible_damage.jpg` to show a visible-damage suggestion.
4. Correct any suggestion using the dropdown fields if needed.
5. Check **I reviewed and verified these three fields**.
6. Select **Use verified information** and download the verified JSON record.

The interface may display **Unable to determine** when confidence is below the configured threshold. This is intentional: the user must select the correct value rather than treating a weak prediction as verified information.

## What to say during the demonstration

The image-information extraction layer proposes three listing fields: material type, component type, and visible condition. Each output has a confidence threshold and remains a suggestion until a human verifies or corrects it. Visible condition means only whether trained visual damage cues are detected; it is not a structural, safety, or reuse-suitability assessment.

## If the browser does not open automatically

Copy the local address printed in the terminal—normally `http://localhost:8501`—and open it in a browser.

## Final offline check

Run this once after setup and before leaving for the presentation:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/test_component.py
PYTHONPATH=src python src/test_multimodel.py
```

If both commands complete without an error, all three model checkpoints can be loaded and the known demonstration images produce their expected outputs.
