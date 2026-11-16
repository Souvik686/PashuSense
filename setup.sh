#!/bin/bash
# ============================================================
# PashuSense — Oracle Cloud VM Setup Script
# Run this ONCE after your first SSH login to the Oracle VM
# ============================================================

set -e   # stop on any error

echo "🔧 Setting up PashuSense on Oracle Cloud..."

# ── 1. System update ─────────────────────────────────────────
sudo apt-get update -y && sudo apt-get upgrade -y

# ── 2. Install Python 3.12 and tools ─────────────────────────
sudo apt-get install -y python3.12 python3.12-venv python3-pip git nginx

# ── 3. Clone your GitHub repo ─────────────────────────────────
# EDIT THIS LINE — replace with your actual GitHub repo URL
REPO_URL="https://github.com/YOUR_USERNAME/pashusense.git"

cd ~
git clone "$REPO_URL" pashusense
cd pashusense

# ── 4. Create virtual environment & install packages ─────────
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ── 5. Install systemd service (keeps app running 24/7) ──────
sudo tee /etc/systemd/system/pashusense.service > /dev/null <<EOF
[Unit]
Description=PashuSense Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/pashusense
Environment="PATH=/home/ubuntu/pashusense/venv/bin"
ExecStart=/home/ubuntu/pashusense/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable pashusense
sudo systemctl start pashusense

# ── 6. Configure Nginx (reverse proxy on port 80) ─────────────
sudo tee /etc/nginx/sites-available/pashusense > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
        client_max_body_size 16M;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/pashusense /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# ── 7. Open firewall ports ───────────────────────────────────
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || sudo apt-get install -y iptables-persistent

echo ""
echo "✅ PashuSense is live!"
echo "   Open in browser: http://$(curl -s ifconfig.me)"
echo ""
echo "   Useful commands:"
echo "   sudo systemctl status pashusense   # check if running"
echo "   sudo journalctl -u pashusense -f   # view live logs"
