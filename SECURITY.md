# Security

## Reporting a vulnerability

Report a vulnerability through GitHub's private **Security → Report a vulnerability** form on this repository, not in a
public issue. Do not post exploit details, tokens or private paths anywhere public.

Include what is affected, how to reproduce it (demo mode is fine), what an attacker could do with it, and the version or
commit you tested. Remove personal paths, host names, addresses and tokens from logs before you send them. Fixes are listed
in [CHANGELOG.md](CHANGELOG.md).

## How the deck is protected

- The deck binds to `127.0.0.1` by default. Every API call needs the token in `~/.config/mcc/token` (created with mode 0600 on first start), and it is compared in constant time.
- INSERT and EJECT can stop running models and delete local copies. Never expose the port directly to the internet. Use an SSH tunnel, a VPN, or an authenticating reverse proxy (see the User Guide, [§9](docs/USER_GUIDE.md#9-remote-access)).
- The deck only talks to the NAS over ssh with BatchMode, and only reads from it (`cat` of the index, `find` for the GGUF list, and `rsync` from the NAS to the GPU box). It never writes to the NAS.
- Deleting a local copy is limited to paths inside `local_dir`. The ids are checked, and the resolved path must stay under that folder.
- vLLM and SGLang run with `--trust-remote-code` only if you set `vllm.trust_remote_code` / `sglang.trust_remote_code` to `true`. Leave it off unless you trust the model repo.
- Recipe lanes run the `start`/`stop` commands in your `recipes.json` as your user. Only list scripts you have read and trust.

## What counts

The deck is built for one trusted user on their own GPU box. `deck.json` and `recipes.json` are trusted configuration:
whoever can edit them can already run commands as you. Reports we especially want: API calls that work without the token,
deleting or reading files outside `local_dir` and the state dir, running commands through request data, cross-site
requests that start an INSERT or EJECT, and anything that makes the deck write to the NAS.
