import fillpdf
from fillpdf import fillpdfs
import sys

# Path to your PDF
input_pdf = "./formular-project/example_data/i-485.pdf"
output_pdf = "./formular-project/example_data/i-485_filled.pdf"


fields = fillpdfs.get_form_fields(input_pdf, sort=False, page_number=None)
print(fields)

sys.exit(0)

# 1. Print / list all form fields and their current values
print("Fields in input PDF:")
fields = fillpdfs.get_form_fields(input_pdf)
for key, val in fields.items():
    print(f"  {key!r}: {val!r}")

# 2. Prepare a dict mapping field names to your desired values
# Use the exact keys printed above:
data_dict = {
    "Family Name (Last Name)": "Doe",
    "Given Name (First Name)": "John",
    "Middle Name (if applicable)": "Alan",
}

# 3. Write filled PDF (not flattened)
fillpdfs.write_fillable_pdf(
    input_pdf, output_pdf, data_dict, flatten=False
)

print(f"\nWritten filled PDF to: {output_pdf}")

# Optionally, you can flatten the filled PDF (make the fields non-editable)
# e.g.:
# fillpdfs.flatten_pdf(output_pdf, "i-485_filled_flattened.pdf")
