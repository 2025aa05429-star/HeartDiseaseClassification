import pandas as pd

# Load results
df = pd.read_csv("models/model_results.csv")

# Round numbers to 3 decimals
df = df.round(3)

# Convert to markdown table
table_md = df.to_markdown(index=False)

# Read existing README
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

start_tag = "<!-- METRICS_TABLE_START -->"
end_tag = "<!-- METRICS_TABLE_END -->"

new_section = f"{start_tag}\n\n{table_md}\n\n{end_tag}"

# Replace placeholder section
import re
readme_updated = re.sub(
    f"{start_tag}.*?{end_tag}",
    new_section,
    readme,
    flags=re.S
)

# Save README
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_updated)

print("README updated with latest metrics!")
