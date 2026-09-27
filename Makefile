RSYNC = rsync -az --delete --exclude __pycache__ --exclude '*.pyc'

deploy-sandbox:
	ssh pl-sandbox 'mkdir -p /opt/plumbline/target'
	$(RSYNC) harness sandboxd pl-sandbox:/opt/plumbline/
	$(RSYNC) target/minigpt pl-sandbox:/opt/plumbline/target/
	scp -q infra/systemd/sandboxd.service pl-sandbox:/etc/systemd/system/
	ssh pl-sandbox 'systemctl daemon-reload && systemctl enable -q sandboxd && systemctl restart sandboxd'

deploy-control:
	ssh pl-control 'mkdir -p /opt/plumbline'
	$(RSYNC) plumbline harness sandboxd infra corpus sabotage Makefile pl-control:/opt/plumbline/
	ssh pl-control "mkdir -p /opt/plumbline/target" && $(RSYNC) --exclude .pytest_cache target/minigpt pl-control:/opt/plumbline/target/
	(git rev-parse HEAD; git diff --quiet HEAD || echo dirty) | tr '\n' ' ' | ssh pl-control 'cat > /opt/plumbline/GIT_SHA'
	rsync -az bakeoff/inputs/ pl-control:/var/lib/plumbline/inputs/
	rsync -az --exclude raw --exclude inputs bakeoff pl-control:/opt/plumbline/   # no --delete: raw outputs live on VM1
	rsync -az results pl-control:/opt/plumbline/
	ssh pl-control 'cp /opt/plumbline/infra/systemd/plumbline-*.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable -q plumbline-judge plumbline-ops && systemctl restart plumbline-judge plumbline-ops'

.PHONY: deploy-sandbox deploy-control
