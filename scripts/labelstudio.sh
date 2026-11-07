#!/usr/bin/env bash
# Label Studio for the pose labelling (Week 10). Khurshid starts it; labelling happens
# in the browser on his laptop through his SSH tunnel (docs_team/labelling_guide.md).
#
#   scripts/labelstudio.sh start  <public|qisens> [opts]   Label Studio on 127.0.0.1:8080 (opts go to
#                                                          label-studio start, e.g. --username/--password
#                                                          for the first login)
#   scripts/labelstudio.sh user   <public|qisens> <email>  add a login (asks for the password)
#   scripts/labelstudio.sh check  <public|qisens>          print the settings in force, start nothing
#
# Everything Label Studio writes (database, logins, label JSON, exports, its config
# and cache folders) stays in data/<area>/labelling/label_studio/. Frames are never
# uploaded: the tasks point at tools/label_frames_server. Analytics, Sentry error
# reports and the PyPI version check are off, signup is closed (logins only through
# "user"), local file import is off, and the server listens on 127.0.0.1 only.
set -euo pipefail
cmd=${1:-}; area=${2:-}
case "$area" in public|qisens) ;; *) echo "area must be public or qisens" >&2; exit 2;; esac
AV=$HOME/AssemblyVision
LS=$AV/envs/labelstudio/bin/label-studio
base=$AV/data/$area/labelling/label_studio
mkdir -p "$base"/{config,cache,data}
chmod 700 "$base"

export XDG_CONFIG_HOME=$base/config XDG_CACHE_HOME=$base/cache XDG_DATA_HOME=$base/data
export LABEL_STUDIO_BASE_DATA_DIR=$base
export LABEL_STUDIO_HOST=http://127.0.0.1:8080
export COLLECT_ANALYTICS=false LABEL_STUDIO_COLLECT_ANALYTICS=false
export SENTRY_DSN= FRONTEND_SENTRY_DSN= LABEL_STUDIO_SENTRY_DSN= LABEL_STUDIO_FRONTEND_SENTRY_DSN=
export LATEST_VERSION_CHECK=false LABEL_STUDIO_LATEST_VERSION_CHECK=false
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_ENABLE_LOCAL_FILES_STORAGE=false LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

case "$cmd" in
  start)
    exec "$LS" start --no-browser --internal-host 127.0.0.1 -p 8080 "${@:3}" ;;
  user)
    email=${3:?email}
    read -r -s -p "password for $email: " pw; echo
    "$LS" user --username "$email" --password "$pw" ;;
  check)
    env | grep -E '^(XDG_|LABEL_STUDIO_|COLLECT_ANALYTICS|SENTRY_DSN|FRONTEND_SENTRY_DSN|LATEST_VERSION_CHECK)' | sort ;;
  *)
    echo "usage: $0 start|user|check <public|qisens> [email]" >&2; exit 2 ;;
esac
