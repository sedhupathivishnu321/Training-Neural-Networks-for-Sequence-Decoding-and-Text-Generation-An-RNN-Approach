import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

# Define a mapping for characters to button sequences
char_to_sequence = {
    "A": "A", "B": "B", "C": "C", "D": "AB", "E": "AC", "F": "BA",
    "G": "CA", "H": "ABC", "I": "BCA", "J": "CAB", "K": "CBA",
    "L": "ABCA", "M": "ABCB", "N": "ABCC", "O": "BCAA", "P": "BCAB",
    "Q": "BCAC", "R": "CABA", "S": "CABB", "T": "CABC", "U": "BACA",
    "V": "BACB", "W": "BACC", "X": "CBAA", "Y": "CBAB", "Z": "CBAC",
    "1": "ABCAA", "2": "ABCAB", "3": "ABCAC", "4": "ABCBA", "5": "ABCBB",
    "6": "ABCBC", "7": "ABCCA", "8": "ABCCB", "9": "ABCCC", "0": "CCCCC",
    " ": "BBBBB"  # Space maps to "BBBBB"
}

def generate_letter_sequences(paragraph):
    # Create DataFrame with each letter sequence as a separate row
    data = {"Sequence": [], "Output": []}

    for word in paragraph.split():
        for char in word:
            sequence = char_to_sequence.get(char.upper(), "")
            data["Sequence"].append(sequence)  # Append character sequence
            data["Output"].append(char.upper())  # Append corresponding letter
            data["Sequence"].append("D")  # Append "D" after each letter
            data["Output"].append("D")  # Append "D" in Output

        # Instead of an empty string, we append "BBBBB" at the end of each word
        data["Sequence"].append("BBBBB")  # Append "BBBBB" to indicate space
        data["Output"].append("")  # Append empty string to keep format

    return pd.DataFrame(data)

def save_to_excel(df):
    # Ask the user where to save the file
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Success", f"Excel file saved at: {file_path}")

def on_generate():
    paragraph = input_text.get("1.0", tk.END).strip()
    if not paragraph:
        messagebox.showwarning("Input Required", "Please enter a paragraph.")
        return

    # Generate sequences DataFrame
    df = generate_letter_sequences(paragraph)

    # Save to Excel
    save_to_excel(df)

# Setup GUI
root = tk.Tk()
root.title("Paragraph to Button Sequence Generator")
root.geometry("500x300")

# Input text box
input_label = tk.Label(root, text="Enter Paragraph:")
input_label.pack(pady=10)
input_text = tk.Text(root, height=10, width=50)
input_text.pack(pady=10)

# Generate button
generate_button = tk.Button(root, text="Generate", command=on_generate)
generate_button.pack(pady=20)

# Run the application
root.mainloop()
