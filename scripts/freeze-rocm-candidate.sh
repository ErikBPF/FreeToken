#!/usr/bin/env bash
set -euo pipefail

output="${1:?usage: freeze-rocm-candidate.sh OUTPUT.tar.gz}"
root="$(git rev-parse --show-toplevel)"
output="$(realpath -m "$output")"
case "$output" in
  "$root"/*) echo "ERROR: output must be outside the source tree" >&2; exit 2 ;;
esac

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
mkdir "$tmp/tree"
git -C "$root" ls-files -co --exclude-standard -z | while IFS= read -r -d '' file; do
  [[ -e "$root/$file" ]] && cp -a --parents "$file" "$tmp/tree"
done
revision="$(git -C "$root" rev-parse HEAD)"
printf '{"schema":1,"revision":"%s"}\n' "$revision" >"$tmp/tree/.freetoken-source.json"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=gnu \
  -C "$tmp/tree" -czf "$tmp/candidate.tar.gz" .
mkdir -p "$(dirname "$output")"
mv "$tmp/candidate.tar.gz" "$output"
sha256sum "$output" | awk '{print $1}' >"$output.sha256"
printf '%s\n' "$(<"$output.sha256")"
