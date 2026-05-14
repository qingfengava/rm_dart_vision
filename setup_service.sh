#!/usr/bin/env bash
# 自动管理 systemd service 文件
# 工作目录 = 脚本所在路径

SERVICE_NAME="dart_vision"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"

ACTION=$1

echo "脚本路径: $SCRIPT_DIR"
echo "工作区路径: $WORK_DIR"
echo "目标 Service 文件: $SERVICE_FILE"
echo

uninstall_service() {

    if [[ ! -f "$SERVICE_FILE" ]]; then
        echo "⚠️  Service 文件不存在，无需卸载。"
        exit 0
    fi

    echo "🛑 停止并卸载 Service..."

    sudo systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
    sudo systemctl disable "${SERVICE_NAME}.service" 2>/dev/null || true
    sudo rm -f "$SERVICE_FILE"

    sudo systemctl daemon-reload

    echo "✅ Service 已成功卸载。"
    exit 0
}

install_service() {
    local force="${2:-}"

    if [[ ! -x "$WORK_DIR/vision" ]]; then
        echo "❌ vision 未编译 (缺少 $WORK_DIR/vision)"
        echo "   请先运行: bash run.sh build"
        exit 1
    fi
    if [[ ! -f "$WORK_DIR/env.sh" ]]; then
        echo "❌ env.sh 不存在 (缺少 $WORK_DIR/env.sh)"
        echo "   请确认 scp.sh 已同步"
        exit 1
    fi

    if [[ -f "$SERVICE_FILE" ]]; then
        if [[ "$force" == "--force" ]]; then
            echo "🔧 强制覆盖已存在的 Service..."
        else
            echo "⚠️  检测到已存在的 Service 文件：$SERVICE_FILE"
            read -p "是否覆盖？(y/N): " confirm
            if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
                echo "🚫 已取消安装。"
                exit 0
            fi
        fi

        echo "🧹 删除旧版本 Service..."
        sudo systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}.service" 2>/dev/null || true
        sudo rm -f "$SERVICE_FILE"
    fi

    echo "✏️  正在生成新的 Service 文件..."

    local runuser="${SUDO_USER:-$USER}"
    sudo usermod -aG video,dialout "$runuser" 2>/dev/null || true
    mkdir -p "$WORK_DIR/data/log" "$WORK_DIR/data/video" 2>/dev/null || true

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Dart Vision Service
After=multi-user.target

[Service]
Type=simple
User=$runuser
WorkingDirectory=$WORK_DIR
EnvironmentFile=-$WORK_DIR/env.sh
ExecStart=$WORK_DIR/vision
Restart=always
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo chmod 644 "$SERVICE_FILE"

    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
    sudo systemctl start "${SERVICE_NAME}.service"

    echo
    echo "✅ Service 已生成并启动成功！"
    echo "🔍 查看状态：sudo systemctl status ${SERVICE_NAME}.service"
    echo "📜 实时日志：journalctl -u ${SERVICE_NAME}.service -f"
}

if [[ -z "$ACTION" ]]; then
    echo "❌ 请输入操作：$0 [install|uninstall|start|stop|restart|status|journal|logs] [--force]"
    exit 1
fi

case "$ACTION" in
    install)
        install_service "$@"
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        sudo systemctl start "${SERVICE_NAME}.service"
        echo "✅ Service 已启动"
        ;;
    stop)
        sudo systemctl stop "${SERVICE_NAME}.service"
        echo "🛑 Service 已停止"
        ;;
    restart)
        sudo systemctl restart "${SERVICE_NAME}.service"
        echo "🔄 Service 已重启"
        ;;
    status)
        sudo systemctl status "${SERVICE_NAME}.service"
        ;;
    journal)
        sudo journalctl -u "${SERVICE_NAME}.service" --no-pager
        ;;
    logs)
        sudo journalctl -u "${SERVICE_NAME}.service" -f
        ;;
    *)
        echo "❌ 参数错误: $0 [install|uninstall|start|stop|restart|status|journal|logs] [--force]"
        exit 1
        ;;
esac
