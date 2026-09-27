RSYNC = rsync -az --delete --exclude __pycache__ --exclude '*.pyc'

deploy-sandbox:
	ssh pl-sandbox 'mkdir -p /opt/plumbline/target'
	$(RSYNC) harness sandboxd pl-sandbox:/opt/plumbline/
	$(RSYNC) target/minigpt pl-sandbox:/opt/plumbline/target/
	scp -q infra/systemd/sandboxd.service pl-sandbox:/etc/systemd/system/
	ssh pl-sandbox 'systemctl daemon-reload && systemctl enable -q sandboxd && systemctl restart sandboxd'

deploy-control:
	ssh pl-control 'mkdir -p /opt/plumbline'
	$(RSYNC) plumbline harness sandboxd infra Makefile pl-control:/opt/plumbline/

.PHONY: deploy-sandbox deploy-control
