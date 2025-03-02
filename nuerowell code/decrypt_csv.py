from cryptography.fernet import Fernet
import base64

def decrypt_csv(encrypted_file_path, decrypted_file_path, key):
    with open(encrypted_file_path, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()
    
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data).decode()

    with open(decrypted_file_path, 'w') as decrypted_file:
        decrypted_file.write(decrypted_data)

if __name__ == "__main__":
    encrypted_file_path = input("Enter the path of the encrypted CSV file: ")
    decrypted_file_path = input("Enter the path to save the decrypted CSV file: ")
    pin = input("Enter the PIN: ")

    if pin == "12345":
        key = base64.urlsafe_b64encode(b'12345'*5)
        decrypt_csv(encrypted_file_path, decrypted_file_path, key)
        print(f"Decrypted file saved to {decrypted_file_path}")
    else:
        print("Invalid PIN!")
