import requests
import os
import hashlib
from tqdm import tqdm
import tarfile
import gzip
import json

# nuScenes credentials
useremail = "gabrielnogueira4014@gmail.com"
password = "Live123!"

output_dir = "data/nuscenes"
region = 'us'  # 'us' or 'asia'

# Mini dataset files
download_files = {
    "v1.0-mini.tgz": "7086c6133a85e5f789c04836b6b545b5",
}


def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }

    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password
        },
        "ClientMetadata": {}
    })

    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )

    if response.status_code == 200:
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print("Authentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("Failed to login. Status code:", response.status_code)
        print(response.text)

    return None


def download_file(url, save_file, md5):
    response = requests.get(url, stream=True)
    if save_file.endswith(".tgz"):
        content_type = response.headers.get('Content-Type', '')
        if content_type == 'application/x-tar':
            save_file = save_file.replace('.tgz', '.tar')
        elif content_type != 'application/octet-stream':
            print("Unknown content type:", content_type)

    if os.path.exists(save_file):
        print(save_file, "already exists, checking MD5...")
        md5obj = hashlib.md5()
        with open(save_file, 'rb') as file:
            for chunk in iter(lambda: file.read(8192), b''):
                md5obj.update(chunk)
        hash_val = md5obj.hexdigest()
        if hash_val != md5:
            print(save_file, "MD5 check failed, downloading again...")
        else:
            print(save_file, "MD5 check passed")
            return save_file

    file_size = int(response.headers.get('Content-Length', 0))
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=os.path.basename(save_file), ascii=True)

    md5obj = hashlib.md5()
    with open(save_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                md5obj.update(chunk)
                file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    hash_val = md5obj.hexdigest()
    if hash_val != md5:
        print(save_file, "MD5 check failed!")
    else:
        print(save_file, "MD5 check passed")

    return save_file


def extract_tgz_to_folder(tgz_file_path, output_folder):
    print(f"Extracting {tgz_file_path} to {output_folder}")
    with gzip.open(tgz_file_path, 'rb') as f_in:
        with tarfile.open(fileobj=f_in, mode='r') as tar:
            tar.extractall(output_folder)


def extract_tar_to_folder(tar_file_path, output_folder):
    print(f"Extracting {tar_file_path} to {output_folder}")
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(output_folder)


def main():
    print("Logging in...")
    bearer_token = login(useremail, password)
    if bearer_token is None:
        print("Login failed, exiting.")
        return

    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

    print("Getting download URLs...")
    download_data = {}
    for filename, md5 in download_files.items():
        api_url = f'https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename}?region={region}&project=nuScenes'

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            print(filename, 'URL obtained successfully')
            download_url = response.json()['url']
            download_data[filename] = [download_url, os.path.join(output_dir, filename), md5]
        else:
            print(f'Request failed: {response.status_code}')
            print(response.text)

    if not download_data:
        print("No files to download, exiting.")
        return

    print("Downloading files...")
    os.makedirs(output_dir, exist_ok=True)

    for output_name, (download_url, save_file, md5) in download_data.items():
        save_file = download_file(download_url, save_file, md5)
        download_data[output_name] = [download_url, save_file, md5]

    print("Extracting files...")
    for output_name, (download_url, save_file, md5) in download_data.items():
        if save_file.endswith(".tgz"):
            extract_tgz_to_folder(save_file, output_dir)
        elif save_file.endswith(".tar"):
            extract_tar_to_folder(save_file, output_dir)
        else:
            print("Unknown file type:", output_name)

    print("Done! nuScenes mini dataset extracted to:", output_dir)


if __name__ == "__main__":
    main()
