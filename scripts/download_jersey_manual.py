import os

from SoccerNet.Downloader import SoccerNetDownloader


def download_jersey_manual():
    downloader = SoccerNetDownloader(LocalDirectory="data/sn_jersey")
    # Using the token from the source code
    token = "ejGBsiDr47cTXnf"
    password = "SoccerNet"

    print(f"Attempting to download train.zip from token {token}...")
    try:
        # path_owncloud needs to be the full URL
        path_owncloud = os.path.join(downloader.OwnCloudServer, "train.zip").replace(' ', '%20').replace('\\', '/')
        downloader.downloadFile(path_owncloud=path_owncloud,
                                path_local=os.path.join(downloader.LocalDirectory, "train.zip"), user=token,
                                password=password)
        print("Download train.zip finished (check if successful)")
    except Exception as e:
        print(f"Error downloading train.zip: {e}")

    print(f"Attempting to download test.zip from token {token}...")
    try:
        path_owncloud = os.path.join(downloader.OwnCloudServer, "test.zip").replace(' ', '%20').replace('\\', '/')
        downloader.downloadFile(path_owncloud=path_owncloud,
                                path_local=os.path.join(downloader.LocalDirectory, "test.zip"), user=token,
                                password=password)
        print("Download test.zip finished (check if successful)")
    except Exception as e:
        print(f"Error downloading test.zip: {e}")


if __name__ == "__main__":
    download_jersey_manual()
