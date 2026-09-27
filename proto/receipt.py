"""Receipts: canonical JSON, ed25519 signature over everything except `signature`."""
import base64, hashlib, json, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization as S


def canonical(obj) -> bytes:
    body = {k: v for k, v in obj.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def keygen():
    k = Ed25519PrivateKey.generate()
    return (k.private_bytes(S.Encoding.PEM, S.PrivateFormat.PKCS8, S.NoEncryption()),
            k.public_key().public_bytes(S.Encoding.Raw, S.PublicFormat.Raw))


def sign(receipt: dict, priv_pem: bytes) -> dict:
    k = S.load_pem_private_key(priv_pem, None)
    pub = k.public_key().public_bytes(S.Encoding.Raw, S.PublicFormat.Raw)
    msg = canonical(receipt)
    receipt["signature"] = {"alg": "ed25519", "key_id": hashlib.sha256(pub).hexdigest()[:16],
                            "public_key": base64.b64encode(pub).decode(),
                            "body_sha256": hashlib.sha256(msg).hexdigest(),
                            "sig": base64.b64encode(k.sign(msg)).decode()}
    return receipt


def verify(receipt: dict, pinned_pub_b64: str | None = None) -> tuple[bool, str]:
    s = receipt.get("signature") or {}
    if pinned_pub_b64 and s.get("public_key") != pinned_pub_b64:
        return False, "public key is not the pinned Plumbline key"
    pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(s["public_key"]))
    try:
        pub.verify(base64.b64decode(s["sig"]), canonical(receipt))
    except Exception:
        return False, "signature does not match body"
    return True, f"valid, key {s['key_id']}, body {s['body_sha256'][:12]}"


if __name__ == "__main__":
    priv, pub = keygen()
    r = sign({"receipt_version": "plumbline/1", "verdict": "BLOCK",
              "checks": [{"family": "ga_invariance", "head": {"metric": 0.75, "threshold": 5.39e-6}}]}, priv)
    print(verify(r, base64.b64encode(pub).decode()))
    r["verdict"] = "PASS"
    print(verify(r, base64.b64encode(pub).decode()))
    try:
        canonical({"x": float("nan")})
    except ValueError as e:
        print("nan rejected:", e)
