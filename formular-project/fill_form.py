from PyPDF2 import PdfReader, PdfWriter

# Load the PDF
reader = PdfReader("./formular-project/example_data/i-485.pdf")
writer = PdfWriter()

# Check available form fields
fields = reader.get_fields()
print("Available fields:")
for field_name in fields:
    print(field_name)

# Copy pages into writer
writer.append_pages_from_reader(reader)

# Fill some fields (replace with real values)
writer.update_page_form_field_values(
    writer.pages[0],
    {
        "form1[0].#subform[0].Pt1Ln1_FamilyName[0]": "DOE",       # Last name
        "form1[0].#subform[0].Pt1Ln1_GivenName[0]": "JOHN",       # First name
        "form1[0].#subform[0].Pt1Ln3_DateofBirth[0]": "01/01/1990"
    }
)

# Save to a new file
with open("./formular-project/example_data/i-485-filled.pdf", "wb") as f:
    writer.write(f)

print("✅ PDF filled and saved as i-485-filled.pdf")
