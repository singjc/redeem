#!/usr/bin/env bash
set -euo pipefail

target="${1:-}"
out_dir="${2:-}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ -z "${target}" ]]; then
  host_triple="$(rustc -vV | sed -n 's/^host: //p')"
  target="${host_triple}"
fi

if [[ -z "${out_dir}" ]]; then
  out_dir="${repo_root}/target/openms-ffi-bundle/${target}"
fi

mkdir -p "${out_dir}/lib"

cargo build --release --package redeem-openms-ffi --target "${target}"

native_static_libs="$(
  cargo rustc --release --package redeem-openms-ffi --target "${target}" -- --print native-static-libs 2>&1 \
    | sed -n 's/^note: native-static-libs: //p' \
    | tail -n 1
)"

if [[ "${target}" == *windows-msvc ]]; then
  built_lib="${repo_root}/target/${target}/release/redeem_openms_ffi.lib"
else
  built_lib="${repo_root}/target/${target}/release/libredeem_openms_ffi.a"
fi

if [[ ! -f "${built_lib}" ]]; then
  echo "error: expected built library at ${built_lib}" >&2
  exit 1
fi

cp "${built_lib}" "${out_dir}/lib/"
printf '%s\n' "${native_static_libs}" > "${out_dir}/lib/redeem-openms-ffi.native-static-libs.txt"
cp "${script_dir}/README.md" "${out_dir}/README.md"
cp "${repo_root}/LICENSE" "${out_dir}/LICENSE"

cat > "${out_dir}/manifest.txt" <<EOF
crate=redeem-openms-ffi
target=${target}
library=$(basename "${built_lib}")
native_static_libs=${native_static_libs}
EOF

echo "created bundle at ${out_dir}"
