from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import csv
#todo:
#load library
#export library 

#Record: Bio record object includes 

def parse_fasta(fasta_filename: str) -> list[SeqRecord]:
    """BioPython SeqIO parsing to SeqRecord objects for compatible filetypes.

    .fasta filetypes recommended.

    Args:
        fasta_filename: directory and filename of a target dataset.

    Returns: 
        Python list of SeqRecord objects.
    """
    seq_record_list = [seq for seq in SeqIO.parse(fasta_filename, 'fasta')]
    return seq_record_list

def is_protein(record) -> bool:
    """Check for protein-exclusive alphabet characters to confirm protein status.
    
    Args:
        record: SeqRecord object.
    
    Returns: 
        True if record.seq contains protein-exclusive characters, otherwise False.  
    """
    ProtAlph = 'DEFHIKLMNPQRSVWY' 
    for i in ProtAlph:
        if i in record.seq:
            return True
    return False        

# TODO: add flags for file-reading support ------------------
def get_id(record_list: list[SeqRecord], sequence: str) -> str:
    for record in record_list:
        if record.seq == sequence:
            return record.id
    return ''

def get_seq(record_list: list[SeqRecord], id: str) -> str:
    for record in record_list:
        if record.id == id:
            return record.seq
    return ''
#------------------------------------------------------------
def csv_to_fasta(
    csv_file,
    fasta_file,
    id_col: int,
    seq_col: int,
    info_col: int = None,
    first_row: int = 0,
    n_rows: int = None
):
    """
    Convert selected rows of a CSV file into FASTA format.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file.

    fasta_file : str
        Path to the output FASTA file.

    id_col : int
        Column index containing the sequence ID.

    seq_col : int
        Column index containing the sequence data.

    info_col : int, optional
        Column index containing extra information (e.g., organism).
        If None, no extra info is added to the FASTA header.

    first_row : int, optional
        Row index to start processing from (0-based). Default is 0.

    n_rows : int, optional
        Number of rows to process starting from `first_row`.
        If None, all rows from `first_row` onward are processed.

    Notes
    -----
    - FASTA header format: `>ID | extra_info`
    - If `info_col` is None, header becomes simply `>ID`.
    - Assumes CSV is comma‑delimited and has no header row unless
      the caller adjusts `first_row` accordingly.
    """

    with open(csv_file, newline='', encoding='utf-8') as infile, \
         open(fasta_file, "w", encoding='utf-8') as outfile:

        reader = csv.reader(infile)

        # Skip rows before first_row
        for _ in range(first_row):
            next(reader, None)

        count = 0
        for row in reader:
            if n_rows is not None and count >= n_rows:
                break

            seq_id = row[id_col]
            sequence = row[seq_col]
            extra_info = row[info_col] if info_col is not None else ""

            header = f">{seq_id}"
            if extra_info:
                header += f" | {extra_info}"

            outfile.write(f"{header}\n{sequence}\n")

            count += 1