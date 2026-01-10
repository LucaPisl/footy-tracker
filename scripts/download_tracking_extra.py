from SoccerNet.Downloader import SoccerNetDownloader


def download_tracking_extra():
    # SoccerNet-Tracking
    downloader = SoccerNetDownloader(LocalDirectory="data/soccernet_tracking")
    print("Attempting to download SoccerNet-Tracking test_labels...")
    try:
        # Trying both passwords just in case
        downloader.downloadDataTask(task="tracking", split=["test_labels"], password="SoccerNet")
        print("Download test_labels successful")
    except Exception as e:
        print(f"Download test_labels failed: {e}")


if __name__ == "__main__":
    download_tracking_extra()
