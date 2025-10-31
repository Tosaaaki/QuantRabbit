.PHONY: help vm-deploy vm-tail vm-logs vm-sql vm-exec gcp-bootstrap

-include scripts/vm.env

PROJECT ?=
ZONE ?=
INSTANCE ?=
ACCOUNT ?=
KEYFILE ?=
USE_IAP ?=
REMOTE_DIR ?=

BRANCH ?= main
VM_SERVICE ?= quantrabbit.service
INSTALL ?= 1

# Compose flags for scripts/vm.sh
GCLOUD_FLAGS = -p $(PROJECT) -z $(ZONE) -m $(INSTANCE) \
  $(if $(ACCOUNT),-A $(ACCOUNT),) \
  $(if $(KEYFILE),-k $(KEYFILE),) \
  $(if $(USE_IAP),-t,) \
  $(if $(REMOTE_DIR),-d $(REMOTE_DIR),)

help:
	@echo "Targets:"
	@echo "  vm-deploy   Deploy branch, install, restart service"
	@echo "  vm-tail     Tail systemd logs (set VM_SERVICE)"
	@echo "  vm-logs     Pull remote logs to ./remote_logs"
	@echo "  vm-sql      Run SQL on remote trades.db"
	@echo "  vm-exec     Run CMD='...' on VM"
	@echo "  gcp-bootstrap  Enable project IAM/APIs/OS Login (set GCP_USER and BILLING)"
	@echo ""
	@echo "Tip: copy scripts/vm.env.example to scripts/vm.env and edit."

vm-deploy:
	@[ -n "$(PROJECT)" ] && [ -n "$(ZONE)" ] && [ -n "$(INSTANCE)" ] || { echo "Set PROJECT, ZONE, INSTANCE (scripts/vm.env)."; exit 1; }
	@echo "Deploying $(BRANCH) to $(INSTANCE)"
	@./scripts/vm.sh $(GCLOUD_FLAGS) deploy -b $(BRANCH) $(if $(INSTALL),-i,) --restart $(VM_SERVICE)

vm-tail:
	@[ -n "$(PROJECT)" ] && [ -n "$(ZONE)" ] && [ -n "$(INSTANCE)" ] || { echo "Set PROJECT, ZONE, INSTANCE (scripts/vm.env)."; exit 1; }
	@./scripts/vm.sh $(GCLOUD_FLAGS) tail -s $(VM_SERVICE)

vm-logs:
	@[ -n "$(PROJECT)" ] && [ -n "$(ZONE)" ] && [ -n "$(INSTANCE)" ] || { echo "Set PROJECT, ZONE, INSTANCE (scripts/vm.env)."; exit 1; }
	@./scripts/vm.sh $(GCLOUD_FLAGS) pull-logs -r $(if $(REMOTE_DIR),$(REMOTE_DIR)/logs,/home/$$USER/QuantRabbit/logs)

vm-sql:
	@[ -n "$(PROJECT)" ] && [ -n "$(ZONE)" ] && [ -n "$(INSTANCE)" ] || { echo "Set PROJECT, ZONE, INSTANCE (scripts/vm.env)."; exit 1; }
	@./scripts/vm.sh $(GCLOUD_FLAGS) sql -f $(if $(REMOTE_DIR),$(REMOTE_DIR)/logs/trades.db,/home/$$USER/QuantRabbit/logs/trades.db) -q "$(or $(Q),SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;)"

vm-exec:
	@[ -n "$(PROJECT)" ] && [ -n "$(ZONE)" ] && [ -n "$(INSTANCE)" ] || { echo "Set PROJECT, ZONE, INSTANCE (scripts/vm.env)."; exit 1; }
	@[ -n "$(CMD)" ] || { echo "Usage: make vm-exec CMD='hostnamectl'"; exit 1; }
	@./scripts/vm.sh $(GCLOUD_FLAGS) exec -- $(CMD)

# Enable project for Quantrabbit (dry-run by default)
# Usage:
#   make gcp-bootstrap PROJECT=quantrabbit GCP_USER=www.tosakiweb.net@gmail.com BILLING=0000-AAAAAA-BBBBBB APPLY=1
gcp-bootstrap:
	@[ -n "$(PROJECT)" ] && [ -n "$(GCP_USER)" ] || { echo "Set PROJECT and GCP_USER. Optional: BILLING, APPLY=1"; exit 1; }
	@./scripts/gcp_enable_project.sh -p $(PROJECT) -u $(GCP_USER) $(if $(BILLING),-b $(BILLING),) $(if $(APPLY),--apply,)
