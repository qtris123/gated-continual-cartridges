"""Generate a QASPER context .txt file for KV cache initialization."""
from cartridges.data.qasper.resources import QASPERResource

OUTPUT_PATH = "qasper_context.txt"

resource = QASPERResource(QASPERResource.Config(topic="question"))
text = resource.to_string()

with open(OUTPUT_PATH, "w") as f:
    f.write(text)

print(f"Wrote {len(text)} chars to {OUTPUT_PATH}")
