import os
import webbrowser
from NorenRestApiPy.NorenApi import NorenApi


CLIENT_ID = os.environ["SHOONYA_CLIENT_ID"]
USER_ID = os.environ["SHOONYA_USER_ID"]

OAUTH_URL = (
    "https://api.shoonya.com/"
    "OAuthlogin/authorize/oauth"
)

REST_HOST = (
    "https://api.shoonya.com/"
    "NorenWClientAPI/"
)

WEBSOCKET_URL = (
    "wss://api.shoonya.com/"
    "NorenWSAPI/"
)


def main():
    print("=" * 60)
    print("ATHENA - SHOONYA OFFICIAL SDK OAUTH TEST")
    print("=" * 60)

    print(f"Client ID : {CLIENT_ID}")
    print(f"User ID   : {USER_ID}")
    print()

    login_url = (
        f"{OAUTH_URL}?client_id={CLIENT_ID}"
    )

    print("Opening Shoonya OAuth login...")
    print()
    print(login_url)
    print()

    webbrowser.open(login_url)

    print("1. Log in on the Shoonya page.")
    print("2. Complete authentication.")
    print("3. The Access Restricted page is expected.")
    print("4. Copy ONLY the value after 'code='.")
    print()

    auth_code = input(
        "Enter FRESH OAuth auth code: "
    ).strip()

    if not auth_code:
        print("[ERROR] No auth code entered.")
        return

    print()
    print("Enter your Shoonya Secret Code.")
    print("It will be visible while typing.")
    print("Do not share it with anyone.")
    print()

    secret_code = input(
        "Secret Code: "
    ).strip()

    if not secret_code:
        print("[ERROR] No Secret Code entered.")
        return

    print()
    print("Creating Shoonya API session...")
    print("Using Shoonya's official SDK method.")
    print()

    try:
        api = NorenApi(
            host=REST_HOST,
            websocket=WEBSOCKET_URL
        )

        result = api.getAccessToken(
            auth_code,
            secret_code,
            CLIENT_ID,
            USER_ID
        )

        print("SDK response received.")
        print()

        if result is None:
            print("[FAILED] SDK returned None.")
            print(
                "Shoonya did not return an OAuth token."
            )
            return

        if not isinstance(result, tuple):
            print(
                "[FAILED] Unexpected SDK response type: "
                f"{type(result).__name__}"
            )
            return

        if len(result) != 4:
            print(
                "[FAILED] Unexpected SDK response length: "
                f"{len(result)}"
            )
            return

        access_token = result[0]
        returned_uid = result[1]
        refresh_token = result[2]
        account_id = result[3]

        print("[SUCCESS] OAuth authentication succeeded.")
        print()
        print(
            "Access token received: YES"
        )
        print(
            f"Returned UID         : {returned_uid}"
        )
        print(
            f"Account ID received   : {account_id}"
        )
        print(
            "Refresh token received: YES"
            if refresh_token
            else "Refresh token received: NO"
        )
        print()
        print(
            "The actual access/refresh tokens "
            "will NOT be displayed."
        )

    except Exception as exc:
        print()
        print("[FAILED] OAuth SDK exception.")
        print(
            f"Exception type: {type(exc).__name__}"
        )
        print(
            f"Exception     : {exc}"
        )

    print()
    print("Authentication test complete.")


if __name__ == "__main__":
    main()