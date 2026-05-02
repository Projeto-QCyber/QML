label_map = {
    'Backdoor': 0,
    'DDoS_HTTP': 1,
    'DDoS_ICMP': 2,
    'DDoS_TCP': 3,
    'DDoS_UDP': 4,
    'Fingerprinting': 5,
    'MITM': 6,
    'Password': 7,
    'Port_Scanning': 8,
    'Ransomware': 9,
    'SQL_injection': 10,
    'Uploading': 11,
    'Vulnerability_scanner': 12,
    'XSS': 13,
    'Others': 14,
    'Normal': 99,
}

def decode_type_attack(array_list):
    inv_label_map = {v: k for k, v in label_map.items()}
    decoded_labels = [inv_label_map[label] for label in array_list]
    print(f"Decoded Ground Truth: {decoded_labels}")
