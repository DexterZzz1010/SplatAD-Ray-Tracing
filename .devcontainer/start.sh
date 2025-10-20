#!/bin/bash
# SplatGut 快速启动脚本

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           SplatGut Docker 快速启动                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker"

# 检查Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker Compose"

# 检查NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  NVIDIA Docker支持可能有问题${NC}"
    echo "   尝试: sudo apt-get install nvidia-docker2"
else
    echo -e "${GREEN}✓${NC} NVIDIA Docker"
fi

# 检查镜像
if ! docker images | grep -q "splatgut.*latest"; then
    echo -e "${RED}❌ 找不到镜像 splatgut:latest${NC}"
    echo "   可用镜像:"
    docker images | grep splatgut || echo "   (无)"
    exit 1
fi
echo -e "${GREEN}✓${NC} SplatGut镜像"

echo
echo "选择启动方式:"
echo "  1) Docker Compose (推荐)"
echo "  2) 纯Docker命令"
echo "  3) 仅验证配置"
echo

read -p "选择 [1-3]: " choice

case $choice in
    1)
        echo
        echo "==== 使用Docker Compose启动 ===="
        
        # 检查配置文件
        if [ ! -f "docker-compose.yml" ]; then
            echo -e "${RED}❌ 找不到 docker-compose.yml${NC}"
            exit 1
        fi
        
        # 启动
        echo "启动容器..."
        docker-compose up -d
        
        echo
        echo -e "${GREEN}✓ 容器已启动${NC}"
        echo
        echo "进入容器:"
        echo "  docker-compose exec splatgut bash"
        echo
        echo "停止容器:"
        echo "  docker-compose down"
        echo
        
        # 询问是否立即进入
        read -p "是否立即进入容器? [y/N]: " enter_now
        if [[ $enter_now =~ ^[Yy]$ ]]; then
            docker-compose exec splatgut bash
        fi
        ;;
        
    2)
        echo
        echo "==== 使用纯Docker命令启动 ===="
        echo
        
        # 获取当前目录
        WORKSPACE=$(pwd)
        
        echo "工作目录: $WORKSPACE"
        echo
        
        # 启动容器
        docker run -it --rm \
            --gpus all \
            --shm-size=12gb \
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
            -e DISPLAY=$DISPLAY \
            -p 7007:7007 \
            -v "$WORKSPACE":/workspace \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -w /workspace \
            --name splatgut_dev \
            splatgut:latest \
            /bin/bash
        ;;
        
    3)
        echo
        echo "==== 验证配置 ===="
        echo
        
        # 检查配置文件
        if [ -f "devcontainer.json" ]; then
            echo -e "${GREEN}✓${NC} devcontainer.json"
        else
            echo -e "${YELLOW}⚠${NC}  devcontainer.json 不存在"
        fi
        
        if [ -f "docker-compose.yml" ]; then
            echo -e "${GREEN}✓${NC} docker-compose.yml"
            echo
            echo "配置内容:"
            cat docker-compose.yml
        else
            echo -e "${RED}❌${NC} docker-compose.yml 不存在"
        fi
        
        echo
        echo "镜像信息:"
        docker images | grep splatgut
        ;;
        
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo
echo "完成!"
