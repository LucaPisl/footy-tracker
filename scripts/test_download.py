from SoccerNet.Downloader import SoccerNetDownloader


def test_download():
    # sn-jersey
    downloader_jersey = SoccerNetDownloader(LocalDirectory="data/sn_jersey_test")
    print("Attempting to download sn-jersey test split...")
    try:
        # Using the password from the issue description
        downloader_jersey.downloadDataTask(task="jersey-2023", split=["test"], password="s0cc3rn3t")
        print("Download successful")
    except Exception as e:
        print(f"Download failed: {e}")


if __name__ == "__main__":
    test_download()
