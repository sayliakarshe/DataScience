import fitz  # PyMuPDF


pdf_file = "./formular-project/example_data/Example_form.pdf"
filled_pdf_file = "./formular-project/example_data/Example_form_filled.pdf"

first_name = "RoRo"
last_name = "Sisi"
dob = "01/01/1990"
address = "123 Main St, Anytown, USA"

def read_pdf_form_and_fill_fields(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get the list of form fields on the page
        form_fields = page.widgets()
        
        print(f"Page {page_num + 1} Form Fields:")
        for field in form_fields:
            # Fill the specific field with the desired value
            print("field_name::", field.field_name)
            if field.field_name == "first_name":
                print(f"Filling field '{field.field_name}' with value '{first_name}'")
                field.field_value = first_name
                field.update()
            elif field.field_name == "last_name":
                print(f"Filling field '{field.field_name}' with value '{last_name}'")
                field.field_value = last_name
                field.update()
            elif field.field_name == "dob":
                print(f"Filling field '{field.field_name}' with value ''{dob}'")
                field.field_value = dob
                field.update()
            elif field.field_name == "address":
                print(f"Filling field '{field.field_name}' with value ' '{address}'")
                field.field_value = address
                field.update()
                
    # Save the filled PDF to a new file
    doc.save(filled_pdf_file)
    print(f"Filled PDF saved as: {filled_pdf_file}")
    
def main():
 read_pdf_form_and_fill_fields(pdf_file)

if __name__ == "__main__":
    main()