# First Open-Source Release Checklist

## 0) Release Goal

发布一个可公开克隆、可运行、无数据泄露风险的首发版本。

## 1) Pre-flight Checks

- [ ] 已确认数据集协议允许公开代码（但不公开数据）
- [ ] 仓库中无账号密钥、token、私密链接
- [ ] 不包含原始数据、特征文件、训练产物、大权重
- [ ] README 安装与运行命令可执行
- [ ] LICENSE 已添加

## 2) Suggested Commit Plan

### Commit 1: open-source bootstrap

```bash
git add README.md LICENSE .gitignore docs/FIRST_RELEASE_COMMIT_CHECKLIST.md
git commit -m "chore: prepare open-source release assets"
```

### Commit 2: core code and docs

```bash
git add src scripts train.py infer.py gradio_app.py realtime_demo.py docs requirements.txt pyproject.toml
git commit -m "feat: release discrete sign language recognition baseline"
```

### Commit 3: final polish (optional)

```bash
git add -A
git commit -m "docs: finalize release notes and repository structure"
```

## 3) Push to GitHub

```bash
git branch -M main
git remote add origin git@github.com:<your-username>/Sign-Language-Trans.git
git push -u origin main
```

## 4) Create First Tag (optional)

```bash
git tag -a v1.0.0 -m "First open-source release"
git push origin v1.0.0
```

