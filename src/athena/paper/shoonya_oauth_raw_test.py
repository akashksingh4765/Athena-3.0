import os
import hashlib
import requests


CLIENT_ID = os.environ["SHOONYA_CLIENT_ID"]
USER_ID = os.environ["SHOONYA_USER_ID"]

GEN_ACCESS_TOKEN_URL = (
    "https://api.shoonya.com/NorenWClientAPI/GenAcsTok"
)


def main():
    print("=" * 60)
    print("ATHENA - SHOONYA RAW OAUTH TOKEN TEST")
    print("=" * 60)

    print(f"Client ID : {CLIENT_ID}")
    print(f"User ID   : {USER_ID}")
    print()

    auth_code = input("Enter FRESH OAuth auth code: ").strip()

    if not auth_code:
        print("[ERROR] No auth code entered.")
        return

    print()
    print("Enter your Shoonya Secret Code.")
    print("WARNING: The Secret Code WILL be visible while typing.")
    print("Make sure nobody else can see your screen.")
    secret_code = input("Secret Code: ").strip()

    if not secret_code:
        print("[ERROR] No Secret Code entered.")
        return

    checksum_input = CLIENT_ID + secret_code + auth_code

    checksum = hashlib.sha256(
        checksum_input.encode("utf-8")
    ).hexdigest()

    payload = (
        '{"code":"'
        + auth_code
        + '","checksum":"'
        + checksum
        + '","uid":"'
        + USER_ID
        + '"}'
    )

    headers = {
        "Content-Type": "text/plain"
    }

    print()
    print("Sending GenAcsTok request...")
    print("No credentials or token will be displayed.")
    print()

    try:
        response = requests.post(
            GEN_ACCESS_TOKEN_URL,
            data="jData=" + payload,
            headers=headers,
            timeout=20,
        )

        print(f"HTTP status: {response.status_code}")
        print(
            "Response content type: "
            f"{response.headers.get('Content-Type')}"
        )
        print()

        try:
            result = response.json()
        except ValueError:
            print("[ERROR] Shoonya returned non-JSON response.")
            print("Response body:")
            print(response.text[:1000])
            return

        print("Shoonya response fields:")

        if isinstance(result, dict):
            print(f"stat       : {result.get('stat')}")
            print(f"emsg       : {result.get('emsg')}")

            if result.get("stat") == "Ok":
                print()
                print("[SUCCESS] OAuth token exchange succeeded.")
                print(f"uid        : {result.get('uid')}")
                print(f"actid      : {result.get('actid')}")
                print()
                print(
                    "Access token received successfully "
                    "but will NOT be displayed."
                )
            else:
                print()
                print("[FAILED] Shoonya rejected the token exchange.")

        else:
            print("[ERROR] Unexpected response structure.")
            print(type(result).__name__)

    except requests.RequestException as exc:
        print("[ERROR] HTTP request failed.")
        print(type(exc).__name__)
        print(str(exc))

    print()
    print("Test complete.")


if __name__ == "__main__":
    main()