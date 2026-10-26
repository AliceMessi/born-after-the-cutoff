from build_release import mask

BS = chr(92)


def test_mask_hides_mentions_emails_and_home_directories():
    text = ("ping @alice, mail bob@example.org\n"
            'File "/home/carol/.venv/lib/x.py", /Users/dave/code, '
            "C:" + BS + "Users" + BS + "Erin" + BS + "proj")
    out = mask(text)
    for name in ("alice", "bob@", "carol", "dave", "Erin"):
        assert name not in out
    assert "/home/[user]/.venv" in out and "C:" + BS + "Users" + BS + "[user]" + BS + "proj" in out
    assert mask("see /usr/home and https://example.org/home/") == "see /usr/home and https://example.org/home/"


def test_mask_redacts_credential_like_strings():
    pem = "-----BEGIN PRIVATE KEY-----" + chr(10) + "MIIEabc" + chr(10) + "-----END PRIVATE KEY-----"
    # fake values are assembled at run time so secret scanners do not flag this file
    fake_key = "sk-" + "abcdefghijklmnop" + "1234"
    fake_aud = "0f1e2d3c" + "4b5a6978" + "8796a5b4" + "c3d2e1f0"
    text = f'key = "{fake_key}"; token_aud={fake_aud}; {pem}; token=short'
    out = mask(text)
    assert fake_key not in out and fake_aud not in out and "MIIEabc" not in out
    assert "token_aud=[redacted]" in out and "[private key removed]" in out and "token=short" in out
    assert mask("api_key=get_the_value_from_env_variable") == "api_key=get_the_value_from_env_variable"
