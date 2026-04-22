# 贡献指南

感谢你对 AIQuant 项目的关注！

## 开发流程

1. **Fork 仓库** 或创建功能分支
2. **提交变更**：遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范
3. **确保测试通过**：运行 `pytest`
4. **提交 Pull Request**：使用 PR 模板

## Commit 规范

```
<type>(<scope>): <subject>

<body>
```

类型：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码逻辑）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具/依赖

## 代码规范

- Python 3.11+
- 使用 Black 格式化（line length 120）
- 使用 Ruff 进行 lint
- 新增功能需附带测试

## 分支策略

- `main`: 生产就绪代码
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 紧急修复
