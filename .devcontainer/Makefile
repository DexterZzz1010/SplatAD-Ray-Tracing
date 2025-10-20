# SplatGut Docker 管理 Makefile
# Linus风格: 简单、直接、好用

.PHONY: help start stop restart shell logs ps clean build verify

# 默认目标
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║           SplatGut Docker 管理命令                       ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "使用方法: make <command>"
	@echo ""
	@echo "命令列表:"
	@echo "  make start    - 启动容器"
	@echo "  make stop     - 停止容器"
	@echo "  make restart  - 重启容器"
	@echo "  make shell    - 进入容器bash"
	@echo "  make logs     - 查看容器日志"
	@echo "  make ps       - 查看容器状态"
	@echo "  make clean    - 清理容器和卷"
	@echo "  make verify   - 验证GPU和环境"
	@echo ""

# 启动容器 (后台运行)
start:
	@echo "🚀 启动SplatGut容器..."
	docker-compose up -d
	@echo "✓ 容器已启动"
	@echo "进入容器: make shell"

# 停止容器
stop:
	@echo "🛑 停止容器..."
	docker-compose down
	@echo "✓ 容器已停止"

# 重启容器
restart:
	@echo "🔄 重启容器..."
	docker-compose restart
	@echo "✓ 容器已重启"

# 进入容器
shell:
	@echo "🐚 进入容器..."
	docker-compose exec splatgut bash

# 查看日志
logs:
	docker-compose logs -f

# 查看容器状态
ps:
	docker-compose ps

# 清理容器和卷
clean:
	@echo "🗑️  清理容器..."
	docker-compose down -v
	@echo "✓ 清理完成"

# 验证环境
verify:
	@echo "🔍 验证环境..."
	@echo ""
	@echo "=== Docker信息 ==="
	docker --version
	docker-compose --version
	@echo ""
	@echo "=== NVIDIA驱动 ==="
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "❌ nvidia-smi 不可用"
	@echo ""
	@echo "=== SplatGut镜像 ==="
	docker images | grep splatgut || echo "❌ 找不到splatgut镜像"
	@echo ""
	@echo "=== 容器状态 ==="
	docker-compose ps || echo "容器未启动"

# GPU测试 (需要容器运行中)
test-gpu:
	@echo "🧪 测试GPU..."
	docker-compose exec splatgut python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || echo "❌ 容器未运行或GPU不可用"

# 快速启动并进入
run: start shell

# 查看资源使用
stats:
	docker stats --no-stream splatgut_splatgut_1 2>/dev/null || docker stats --no-stream

# X11权限 (如果GUI不工作)
fix-x11:
	@echo "🔧 修复X11权限..."
	xhost +local:docker
	@echo "✓ X11权限已设置"
