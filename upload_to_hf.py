"""
1-Click Deploy to Hugging Face Spaces (24/7 Free Cloud Hosting)
Usage:
  python upload_to_hf.py --repo your-username/stylefinder --token hf_yourTokenHere
"""
import argparse, os, sys
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload StyleFinder to Hugging Face Spaces")
    parser.add_argument("--repo", required=True, help="Your HF Space repo id (e.g. username/stylefinder)")
    parser.add_argument("--token", required=True, help="Your Hugging Face write token (from https://huggingface.co/settings/tokens)")
    parser.add_argument("--dir", default="deploy_package", help="Folder containing deployment files")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found. Please run: python prepare_cloud_package.py first.")
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"1. Creating / checking Space: {args.repo} ...")
    try:
        api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="docker", exist_ok=True)
        print("   Space created / verified.")
    except Exception as e:
        print(f"   Note on repo creation: {e}")

    print(f"2. Uploading files from {args.dir} to Space {args.repo}...")
    print("   This may take a few minutes depending on your internet upload speed (~2 GB).")
    try:
        api.upload_folder(
            folder_path=args.dir,
            repo_id=args.repo,
            repo_type="space",
            commit_message="Deploy StyleFinder 24/7 on Hugging Face Spaces"
        )
        print("\n=======================================================")
        print(f"[SUCCESS] Deployed successfully!")
        print(f"Your 24/7 public link is: https://huggingface.co/spaces/{args.repo}")
        print(f"Direct app link: https://{args.repo.replace('/', '-')}.hf.space")
        print("=======================================================")
    except Exception as e:
        print(f"\nUpload error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

