import datetime

# subs is a dictionary of replacements to be done on string s
def replace(s, subs):
    for old, new in subs.items():
        s = s.replace(old, new)
    return s

# ISO timestamp sanitized for file names
def file_timestamp():
    iso_timestamp = datetime.datetime.utcnow().isoformat()
    return replace(iso_timestamp, {':': '_', '.': '_', '+': '_', 'T': '_'}) + 'UTC'
