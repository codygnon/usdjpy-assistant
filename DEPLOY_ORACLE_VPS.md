# Deploy USDJPY Assistant on Oracle Cloud VPS

Run the web app and trading loop 24/7 on an Oracle Cloud VM. One URL for you and your uncle; the loop keeps running while you sleep.

## Order of operations

1. **Add deploy files** (already in this repo): `deploy/usdjpy-api.service`, `deploy/usdjpy-loop.service`, `deploy/setup_vps.sh`.
2. **Commit and push** (see bottom of this file).
3. **Oracle Cloud**: Create account → Create VM → Open port 8000.
4. **SSH into the VM** and clone repo, copy profiles, run `./deploy/setup_vps.sh`.
5. **Open** `http://<PUBLIC_IP>:8000` in your browser.

---

## 1. Oracle Cloud (do this yourself)

### Create account

- Go to [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/).
- Sign up (email, password, region). Add a payment method (free tier does not charge unless you upgrade).

### Create a VM instance

1. Console → **Compute** → **Instances** → **Create instance**.
2. **Name:** e.g. `usdjpy-assistant`.
3. **Image:** Canonical Ubuntu 22.04.
4. **Shape:** **VM.Standard.E2.1.Micro** (Always Free: 1 OCPU, 1 GB RAM).
5. **Networking:** Create new VCN if needed; **Assign a public IPv4 address** = checked.
6. **SSH keys:** Generate a key pair and download the private key (e.g. `key.key`). Store it somewhere safe.
7. Click **Create**. Wait until state is **Running**. Note the **Public IP address**.

### Open port 8000

1. **Networking** → **Virtual cloud networks** → click your VCN → **Security Lists** → **Default Security List**.
2. **Add Ingress Rule:**
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: TCP
   - Destination port range: `8000`
3. Save.

---

## 2. First-time setup on the VPS (you run these)

### SSH in

```bash
chmod 600 /path/to/key.key
ssh -i /path/to/key.key ubuntu@<PUBLIC_IP>
```

(If the image is Oracle Linux, use `opc` instead of `ubuntu`.)

### Clone repo and create data dirs

```bash
sudo apt-get update && sudo apt-get install -y git
sudo mkdir -p /opt && sudo chown "$USER" /opt
git clone https://github.com/YOUR_USERNAME/usdjpy_assistant.git /opt/usdjpy_assistant
cd /opt/usdjpy_assistant
mkdir -p data/profiles/v1 data/logs
```

Replace `YOUR_USERNAME/usdjpy_assistant` with your actual repo URL. If the repo is private, use a personal access token or SSH.

### Copy your profile(s) from your Mac to the VPS

From your **local machine** (new terminal), with the VM’s public IP and key path:

```bash
scp -i /path/to/key.key /Users/codygnon/Documents/usdjpy_assistant/profiles/v1/*.json ubuntu@<PUBLIC_IP>:/opt/usdjpy_assistant/data/profiles/v1/
```

### Run the setup script on the VPS

Back in the SSH session:

```bash
cd /opt/usdjpy_assistant
chmod +x deploy/setup_vps.sh
./deploy/setup_vps.sh
```

This installs Python, creates the venv, installs dependencies, and starts the API and loop as systemd services.

### Check services

```bash
sudo systemctl status usdjpy-api usdjpy-loop
```

Open in a browser: **http://<PUBLIC_IP>:8000**. The loop is already running (no need to click “Start Loop”).

---

## 3. Which profile does the loop use?

The loop runs **one** profile 24/7. By default it’s `data/profiles/v1/cody_demo.json`. To use `uncle_demo.json` instead:

1. Edit `deploy/usdjpy-loop.service`: change `cody_demo.json` to `uncle_demo.json` in the `ExecStart` line.
2. On the VPS:
   ```bash
   sudo sed -i 's/cody_demo.json/uncle_demo.json/' /etc/systemd/system/usdjpy-loop.service
   sudo systemctl daemon-reload
   sudo systemctl restart usdjpy-loop
   ```

---

## 4. Useful commands on the VPS

| Task | Command |
|------|--------|
| Status | `sudo systemctl status usdjpy-api usdjpy-loop` |
| Logs (live) | `sudo journalctl -u usdjpy-api -u usdjpy-loop -f` |
| Restart API | `sudo systemctl restart usdjpy-api` |
| Restart loop | `sudo systemctl restart usdjpy-loop` |
| Stop loop | `sudo systemctl stop usdjpy-loop` |
| Update app (after git pull) | `cd /opt/usdjpy_assistant && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart usdjpy-api usdjpy-loop` |

---

## Commit and push (after adding deploy files)

From your project directory on your Mac:

```bash
cd /Users/codygnon/Documents/usdjpy_assistant
git add deploy/ DEPLOY_ORACLE_VPS.md
git status
git commit -m "Add Oracle Cloud VPS deploy: systemd units and setup script"
git push
```

If your branch doesn’t track a remote yet:

```bash
git push -u origin main
```

(Use `master` if that’s your default branch.)

After pushing, clone the repo on the VPS (or `git pull` if already cloned) and run `./deploy/setup_vps.sh` as in section 2.
