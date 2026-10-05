# Security

- The deck binds to `127.0.0.1` by default. Every API call needs the token in `~/.config/mcc/token` (created with mode 0600 on first start), and it is compared in constant time.
- INSERT and EJECT can stop running models and delete local copies. Never expose the port directly to the internet. Use an SSH tunnel, a VPN, or an authenticating reverse proxy (see the User Guide, §8).
- The deck only talks to the NAS over ssh with BatchMode, and only reads from it (`cat` of the index, `find` for the GGUF list, and `rsync` from the NAS to the GPU box). It never writes to the NAS.
- Deleting a local copy is limited to paths inside `local_dir`. The ids are checked, and the resolved path must stay under that folder.
- vLLM runs with `--trust-remote-code` only if you set `vllm.trust_remote_code: true`. Leave it off unless you trust the model repo.

Report a vulnerability through GitHub's private **Security → Report a vulnerability** form on this repository, not in a public issue.
