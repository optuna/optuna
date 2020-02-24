MAKEFILE_PATH := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_PATH))

# "-v" option can not be used in circleci. Instead, this is needed.
.PHONY: _volume
_volume:
	docker rm -f myvolume || echo
	docker create -v /project --name myvolume alpine:3.4 /bin/true
	docker cp $(PROJECT_ROOT)/. myvolume:/project/

.PHONY: black
black: _volume
	docker pull unibeautify/black
	docker run -w /project --volumes-from myvolume unibeautify/black /project --line-length 99 --exclude="docs|tests|examples"
	docker cp myvolume:/project/. $(PROJECT_ROOT)/.
