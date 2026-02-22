import logging
from ex_router.traffic_control import get_routing_decision

# Setup verbose console logging for the test
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)

print("\n--- Testing Phase 2 Routing Logic ---\n")

# Scenario 1: The exact edge case from your previous log
print("Scenario 1: High confidence title, but empty text.")
get_routing_decision(element_type="title", confidence=1.0, base_text="")

# Scenario 2: Perfect, clean paragraph
print("\nScenario 2: Standard paragraph with good OCR.")
get_routing_decision(element_type="text", confidence=0.98, base_text="This is a clean paragraph of text.")

# Scenario 3: Messy, low-confidence text (like bad handwriting)
print("\nScenario 3: Messy text with low OCR confidence.")
get_routing_decision(element_type="text", confidence=0.65, base_text="Th1s 1s h4rd t0 r3ad.")

# Scenario 4: A standard table
print("\nScenario 4: A standard financial table.")
get_routing_decision(element_type="table", confidence=0.95, base_text="Col1 Col2 \n 100 200")

# Scenario 5: A complex chart
print("\nScenario 5: A visual bar chart.")
get_routing_decision(element_type="figure", confidence=0.88, base_text="")

print("\n--- Routing Tests Complete ---\n")