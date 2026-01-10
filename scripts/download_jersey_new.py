from SoccerNet.Downloader import SoccerNetDownloader


def download_jersey():
    # sn-jersey
    downloader_jersey = SoccerNetDownloader(LocalDirectory="data/sn_jersey")
    for split in ["train", "test", "challenge"]:
        print(f"Attempting to download sn-jersey {split} split...")
        try:
            downloader_jersey.downloadDataTask(task="jersey-2023", split=[split], password="SoccerNet")
            print(f"Download {split} successful")
        except Exception as e:
            print(f"Download {split} failed: {e}")


if __name__ == "__main__":
    download_jersey()
