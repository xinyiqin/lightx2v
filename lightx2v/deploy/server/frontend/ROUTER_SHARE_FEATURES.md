# 路由和分享功能实现说明

## 功能概述

本次更新为项目添加了完整的路由支持和分享功能，包括：

1. **任务详情路由支持** - 每个任务都有独立的URL
2. **页面路由支持** - 所有页面状态都能通过URL保存和恢复
3. **分享功能** - 用户可以分享生成的视频到社交平台

## 实现的功能

### 1. 任务详情路由 (`/task/:taskId`)

- **URL格式**: `/task/{taskId}`
- **功能**: 直接通过URL访问特定任务的详情
- **实现**: 
  - 在 `router/index.js` 中添加了 `TaskDetail` 路由
  - 在 `Projects.vue` 中添加了路由监听
  - 在 `other.js` 中实现了 `openTaskFromRoute` 函数

### 2. 模板详情路由 (`/template/:templateId`)

- **URL格式**: `/template/{templateId}`
- **功能**: 直接通过URL访问特定模板的详情
- **实现**: 
  - 在 `router/index.js` 中添加了 `TemplateDetail` 路由
  - 在 `Inspirations.vue` 中添加了路由监听
  - 在 `other.js` 中实现了 `openTemplateFromRoute` 函数

### 3. 页面路由支持

#### Generate页面 (`/generate`)
- **支持的URL参数**:
  - `taskType`: 任务类型
  - `model`: 选择的模型
  - `expanded`: 是否展开创建区域
- **示例URL**: `/generate?taskType=t2v&model=model1&expanded=true`

#### Projects页面 (`/projects`)
- **支持的URL参数**:
  - `search`: 搜索关键词
  - `status`: 状态筛选 (ALL, SUCCEED, RUNNING, FAILED)
  - `page`: 当前页码
- **示例URL**: `/projects?search=test&status=SUCCEED&page=2`

#### Inspirations页面 (`/inspirations`)
- **支持的URL参数**:
  - `search`: 搜索关键词
  - `category`: 分类筛选
  - `page`: 当前页码
- **示例URL**: `/inspirations?search=creative&category=animation&page=3`

### 4. 分享功能

#### 任务分享链接生成
- **功能**: 生成可分享的任务链接
- **实现**: `generateShareUrl(taskId)` 函数
- **URL格式**: `{domain}/task/{taskId}`

#### 模板分享链接生成
- **功能**: 生成可分享的模板链接
- **实现**: `generateTemplateShareUrl(templateId)` 函数
- **URL格式**: `{domain}/template/{templateId}`

#### 社交平台分享
支持的平台：
- Twitter
- Facebook
- LinkedIn
- WhatsApp
- Telegram
- 微博

#### 分享按钮位置
- **任务详情弹窗**: 分享区域和操作按钮组
- **模板详情弹窗**: 头部分享按钮和详细分享区域
- **成功任务**: 操作按钮组中的分享按钮
- **复制链接**: 所有分享功能都支持一键复制

## 技术实现

### 路由监听
```javascript
// 监听路由参数变化
watch(() => route.query, (newQuery) => {
    // 同步URL参数到组件状态
}, { immediate: true })

// 监听组件状态变化，同步到URL
watch([componentState], () => {
    router.replace({ query })
})
```

### 分享功能
```javascript
// 生成分享链接
const generateShareUrl = (taskId) => {
    const baseUrl = window.location.origin;
    return `${baseUrl}/task/${taskId}`;
}

// 复制到剪贴板
const copyShareLink = async (taskId) => {
    const shareUrl = generateShareUrl(taskId);
    await navigator.clipboard.writeText(shareUrl);
}

// 分享到社交平台
const shareToSocial = (taskId, platform) => {
    // 根据不同平台生成分享URL
}
```

## 使用方法

### 1. 直接访问任务
用户可以通过以下URL直接访问任务：
```
https://yourdomain.com/task/12345
```

### 2. 直接访问模板
用户可以通过以下URL直接访问模板：
```
https://yourdomain.com/template/67890
```

### 3. 分享任务
在任务详情弹窗中：
1. 点击"复制分享链接"按钮复制链接
2. 点击社交平台按钮分享到对应平台

### 4. 分享模板
在模板详情弹窗中：
1. 点击头部的"分享模板"按钮快速复制链接
2. 在详细分享区域选择社交平台分享
3. 支持复制链接到剪贴板

### 5. 保存页面状态
页面的所有状态（搜索、筛选、分页等）都会自动保存到URL中，用户可以通过URL直接访问特定状态的页面。

## 注意事项

1. **Node.js版本**: 当前项目需要Node.js 20.19+或22.12+版本
2. **路由守卫**: 所有路由都需要用户登录
3. **URL同步**: 组件状态和URL参数会自动双向同步
4. **分享权限**: 只有成功完成的任务才能被分享

## 文件修改清单

- `src/router/index.js` - 添加任务详情路由、模板详情路由和props支持
- `src/utils/other.js` - 添加分享功能和路由相关函数
- `src/components/Projects.vue` - 添加路由监听和分享功能导入
- `src/components/TaskDetails.vue` - 添加分享按钮和功能
- `src/components/Generate.vue` - 添加路由监听
- `src/components/Inspirations.vue` - 添加路由监听和模板路由支持
- `src/components/TemplateDetails.vue` - 添加完整的分享功能

所有功能已经实现并测试通过。
