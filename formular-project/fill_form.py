from pdfrw import PdfReader, PdfWriter, PdfDict

# Load the blank form
template_pdf = PdfReader("./formular-project/example_data/i-485.pdf")

# Map of field names to values
# (You may need to adjust the keys depending on the actual field names in the PDF)
data_dict = {
    'Family Name (Last Name)': 'Doe',
    'Given Name (First Name)': 'John',
    'Middle Name (if applicable)': 'Alan'
}

# Loop over pages and fill fields
for page in template_pdf.pages:
    if '/Annots' in page:
        for annotation in page['/Annots']:
            field = annotation.get_object()
            if field['/Subtype'] == '/Widget' and '/T' in field:
                key = field['/T'][1:-1]  # remove parentheses
                if key in data_dict:
                    field.update(
                        PdfDict(V='{}'.format(data_dict[key]))
                    )

# Write the filled form to a new PDF
PdfWriter().write("./formular-project/example_data/i-485_filled.pdf", template_pdf)
