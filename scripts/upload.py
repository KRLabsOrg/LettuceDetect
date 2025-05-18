import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained model and tokenizer to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path to the saved model directory (contains model and tokenizer files).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target repository id on Hugging Face (e.g., KRLabsOrg/lettucedect-base-modernbert-en-v1).",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Include this flag to use your Hugging Face authentication token (if not already set up).",
    )
    args = parser.parse_args()

    print(f"Loading model and tokenizer from {args.model_path} ...")
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Uploading model to Hugging Face Hub at repo: {args.repo_id} ...")
    model.push_to_hub(args.repo_id, use_auth_token=args.use_auth_token)
    tokenizer.push_to_hub(args.repo_id, use_auth_token=args.use_auth_token)
    print("Upload complete!")


if __name__ == "__main__":
    main()
