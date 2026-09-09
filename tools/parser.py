from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
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
        Python list of ReqRecord objects.
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