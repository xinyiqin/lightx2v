<script setup>
import { showTaskDetailModal,
        modalTask,
        closeTaskDetailModal,
        cancelTask,
        reuseTask,
        downloadFile,
        deleteTask,
        getTaskTypeName,
        formatTime,
        getTaskStatusDisplay,
        getStatusTextClass,
        getProgressTitle,
        getProgressInfo,
        getOverallProgress,
        getSubtaskStatusText,
        getTaskFailureInfo,
        selectedTaskFiles,
        generateShareUrl,
        copyShareLink,
        shareToSocial,
        copyPrompt
         } from '../utils/other'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { ref, onMounted, onUnmounted } from 'vue'
const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()

// 添加响应式变量
const showDetails = ref(false)

// 获取图片素材
const getImageMaterials = () => {
    if (!modalTask.value?.inputs) return []
    const imageMaterials = Object.entries(modalTask.value.inputs).filter(([name, input]) => 
        name.includes('image') && selectedTaskFiles.value.inputs[name] && selectedTaskFiles.value.inputs[name].url
    ).map(([name, input]) => [name, selectedTaskFiles.value.inputs[name].url])
    return imageMaterials
}

// 获取音频素材
const getAudioMaterials = () => {
    if (!modalTask.value?.inputs) return []
    const audioMaterials = Object.entries(modalTask.value.inputs).filter(([name, input]) => 
        name.includes('audio') && selectedTaskFiles.value.inputs[name] && selectedTaskFiles.value.inputs[name].url
    ).map(([name, input]) => [name, selectedTaskFiles.value.inputs[name].url])
    return audioMaterials
}

// 路由关闭功能
const closeWithRoute = () => {
    closeTaskDetailModal()
    // 如果是从路由进入的，可以返回到上一页
    if (window.history.length > 1) {
        router.go(-1)
    } else {
        router.push('/')
    }
}

// 键盘事件处理
const handleKeydown = (event) => {
    if (event.key === 'Escape' && showTaskDetailModal.value) {
        closeWithRoute()
    }
}

// 生命周期钩子
onMounted(() => {
    document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
})
</script>
<template>
            <!-- 任务详情弹窗 -->
            <div v-cloak>
                <div v-if="showTaskDetailModal"
                    class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
                    @click="closeWithRoute">
                    <!-- 任务完成时的大弹窗 -->
                    <div v-if="modalTask?.status === 'SUCCEED'"
                        class="landing-page" @click.stop>
                        <!-- 弹窗头部 -->
                        <div class="modal-header">
                            <h3 class="modal-title">
                                <i class="fas fa-info-circle mr-2"></i>
                                {{ t('taskDetails') }}
                            </h3>
                            <div class="header-actions">
                                <button @click="closeWithRoute" class="action-button back-button" :title="t('back')">
                                    <i class="fas fa-arrow-left"></i>
                                </button>
                                <button @click="closeWithRoute" class="action-button close-button" :title="t('close')">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                        </div>

                        <!-- 主要内容区域 -->
                        <div class="main-content main-scrollbar overflow-y-auto">
                            <!-- 分享内容 -->
                            <div class="content-grid">
                                <!-- 左侧视频区域 -->
                                <div class="video-section">
                                    <div class="video-container">
                                        <!-- 视频播放器 -->
                                        <video
                                            v-if="selectedTaskFiles.outputs.output_video && selectedTaskFiles.outputs.output_video.url"
                                            :src="selectedTaskFiles.outputs.output_video.url"
                                            class="video-player"
                                            controls
                                            autoplay
                                            muted
                                            loop
                                            preload="metadata"
                                            @loadstart="onVideoLoadStart"
                                            @canplay="onVideoCanPlay"
                                            @error="onVideoError">
                                            {{ t('browserNotSupported') }}
                                        </video>
                                        <div v-else class="video-placeholder">
                                            <div class="loading-spinner">
                                                <i class="fas fa-video"></i>
                                            </div>
                                            <p class="loading-text">{{ t('videoNotAvailable') }}</p>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 -->
                                <div class="info-section">
                                    <div class="info-content">
                                        <!-- 标题 -->
                                        <h1 class="main-title">
                                            {{ t('taskDetails') }}
                                        </h1>
                                        
                                        <!-- 描述 -->
                                        <p class="main-description">
                                            {{ t('taskCompletedSuccessfully') }}
                                        </p>
                                        
                                        <div class="features-list justify-between">
                                            <div class="feature-item"">
                                                <i class="fas fa-rocket feature-icon"></i>
                                                <span>getTaskTypeName(modalTask)</span>
                                            </div>
                                            <div class="feature-item cursor-pointer"">
                                                <i class="fas fa-bolt feature-icon"></i>
                                                <span>modalTask.model_cls</span>
                                            </div>
                                        </div>
                                        
                                        <!-- 操作按钮 -->
                                        <div class="action-buttons">
                                            <button @click="reuseTask(modalTask); closeTaskDetailModal()" class="primary-button">
                                                <i class="fas fa-magic mr-2"></i>
                                                {{ t('reuseTask') }}
                                            </button>
                                            
                                            <button v-if="selectedTaskFiles.outputs.output_video && selectedTaskFiles.outputs.output_video.url"
                                                    @click="downloadFile(selectedTaskFiles.outputs.output_video)" class="secondary-button">
                                                <i class="fas fa-download mr-2"></i>
                                                {{ t('downloadVideo') }}
                                            </button>
                                            
                                            <button v-if="modalTask?.status === 'SUCCEED'"
                                                    @click="copyShareLink(modalTask.task_id, 'task')" class="secondary-button">
                                                <i class="fas fa-share-alt mr-2"></i>
                                                {{ t('copyShareLink') }}
                                            </button>
                                            
                                            <button @click="showDetails = !showDetails" class="secondary-button">
                                                <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="mr-2"></i>
                                                {{ showDetails ? t('hideDetails') : t('showDetails') }}
                                            </button>
                                        </div>
                                        
                                        <!-- 技术信息 -->
                                        <div class="tech-info">
                                            <p class="tech-text">{{ t('poweredByLightX2V') }}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 详细信息面板 -->
                        <div v-if="showDetails && modalTask" class="details-panel">
                            <div class="details-content">
                                <!-- 输入素材标题 -->
                                <div class="materials-header">
                                    <h2 class="materials-title">
                                        <i class="fas fa-upload mr-2"></i>
                                        {{ t('inputMaterials') }}
                                    </h2>
                                </div>
                                
                                <!-- 三个并列的分块卡片 -->
                                <div class="materials-cards">
                                    <!-- 图片卡片 -->
                                    <div class="material-card">
                                        <div class="card-header">
                                            <i class="fas fa-image card-icon"></i>
                                            <h3 class="card-title">{{ t('image') }}</h3>
                                            <div class="card-actions">
                                                <button v-if="getImageMaterials().length > 0" 
                                                        @click="downloadFile(selectedTaskFiles.inputs[getImageMaterials()[0][0]])"
                                                        class="action-btn download-btn" 
                                                        :title="t('download')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="getImageMaterials().length > 0" class="image-grid">
                                                <div v-for="[inputName, url] in getImageMaterials()" :key="inputName" class="image-item">
                                                    <div class="image-container">
                                                        <img :src="url" :alt="inputName" class="image-preview">
                                                    </div>
                                                </div>
                                            </div>
                                            <div v-else class="empty-state">
                                                <i class="fas fa-image empty-icon"></i>
                                                <p class="empty-text">{{ t('noImage') }}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 音频卡片 -->
                                    <div class="material-card">
                                        <div class="card-header">
                                            <i class="fas fa-music card-icon"></i>
                                            <h3 class="card-title">{{ t('audio') }}</h3>
                                            <div class="card-actions">
                                                <button v-if="getAudioMaterials().length > 0" 
                                                        @click="downloadFile(selectedTaskFiles.inputs[getAudioMaterials()[0][0]])"
                                                        class="action-btn download-btn" 
                                                        :title="t('download')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="getAudioMaterials().length > 0" class="audio-list">
                                                <div v-for="[inputName, url] in getAudioMaterials()" :key="inputName" class="audio-item">
                                                    <audio :src="url" controls class="audio-player"></audio>
                                                </div>
                                            </div>
                                            <div v-else class="empty-state">
                                                <i class="fas fa-music empty-icon"></i>
                                                <p class="empty-text">{{ t('noAudio') }}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 提示词卡片 -->
                                    <div class="material-card">
                                        <div class="card-header">
                                            <i class="fas fa-file-alt card-icon"></i>
                                            <h3 class="card-title">{{ t('prompt') }}</h3>
                                            <div class="card-actions">
                                                <button v-if="modalTask?.params?.prompt" 
                                                        @click="copyPrompt(modalTask?.params?.prompt)"
                                                        class="action-btn copy-btn" 
                                                        :title="t('copy')">
                                                    <i class="fas fa-copy"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="modalTask?.params?.prompt" class="prompt-content">
                                                <p class="prompt-text">{{ modalTask.params.prompt }}</p>
                                            </div>
                                            <div v-else class="empty-state">
                                                <i class="fas fa-file-alt empty-icon"></i>
                                                <p class="empty-text">{{ t('noPrompt') }}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 其他状态的弹窗 -->
                    <div v-else class="landing-page" @click.stop>
                        <!-- 弹窗头部 -->
                        <div class="modal-header">
                            <h3 class="modal-title">
                                <i class="fas fa-info-circle mr-2"></i>
                                {{ t('taskDetails') }}
                            </h3>
                            <div class="header-actions">
                                <button @click="closeWithRoute" class="action-button back-button" :title="t('back')">
                                    <i class="fas fa-arrow-left"></i>
                                </button>
                                <button @click="closeTaskDetailModal" class="action-button close-button" :title="t('close')">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                        </div>

                        <!-- 主要内容区域 -->
                        <div class="main-content main-scrollbar overflow-y-auto">
                            <div class="content-grid">
                                <!-- 左侧占位图区域 -->
                                <div class="video-section">
                                    <div class="video-container">
                                        <!-- 根据状态显示不同的占位图 -->
                                        <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)" class="video-placeholder">
                                            <div class="loading-spinner">
                                                <i class="fas fa-spinner fa-spin"></i>
                                            </div>
                                            <p class="loading-text">{{ t('videoGenerating') }}...</p>
                                        </div>
                                        <div v-else-if="modalTask?.status === 'FAILED'" class="video-placeholder error-placeholder">
                                            <div class="error-icon">
                                                <i class="fas fa-exclamation-triangle"></i>
                                            </div>
                                            <p class="loading-text">{{ t('videoGeneratingFailed') }}</p>
                                        </div>
                                        <div v-else-if="modalTask?.status === 'CANCEL'" class="video-placeholder cancel-placeholder">
                                            <div class="cancel-icon">
                                                <i class="fas fa-ban"></i>
                                            </div>
                                            <p class="loading-text">{{ t('taskCancelled') }}</p>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 -->
                                <div class="info-section">
                                    <div class="info-content">
                                        <!-- 标题 -->
                                        <h1 class="main-title">
                                            {{ t('taskDetails') }}
                                        </h1>
                                        
                                        <!-- 描述 -->
                                        <p class="main-description">
                                            <span v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)">
                                                {{ t('aiIsGeneratingYourVideo') }}...
                                            </span>
                                            <span v-else-if="modalTask?.status === 'FAILED'">
                                                {{ t('sorryYourVideoGenerationTaskFailed') }}
                                            </span>
                                            <span v-else-if="modalTask?.status === 'CANCEL'">
                                                {{ t('thisTaskHasBeenCancelledYouCanRegenerateOrViewTheMaterialsYouUploadedBefore') }}
                                            </span>
                                        </p>
                                        
                                        <!-- 操作按钮 -->
                                        <div class="action-buttons">
                                            <!-- 进行中状态 -->
                                            <button v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)"
                                                    @click="cancelTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="primary-button">
                                                <i class="fas fa-times mr-2"></i>
                                                {{ t('cancelTask') }}
                                            </button>
                                            
                                            <!-- 失败状态 -->
                                            <button v-if="modalTask?.status === 'FAILED'"
                                                    @click="resumeTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="primary-button">
                                                <i class="fas fa-redo mr-2"></i>
                                                {{ t('retryTask') }}
                                            </button>
                                            
                                            <!-- 取消状态 -->
                                            <button v-if="modalTask?.status === 'CANCEL'"
                                                    @click="resumeTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="primary-button">
                                                <i class="fas fa-redo mr-2"></i>
                                                {{ t('regenerateTask') }}
                                            </button>
                                            
                                            <!-- 通用按钮 -->
                                            <button v-if="['SUCCEED', 'FAILED', 'CANCEL','CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)"
                                                    @click="reuseTask(modalTask); closeTaskDetailModal()" 
                                                    class="secondary-button">
                                                <i class="fas fa-copy mr-2"></i>
                                                {{ t('reuseTask') }}
                                            </button>
                                            
                                            <button v-if="modalTask?.status === 'SUCCEED'"
                                                    @click="copyShareLink(modalTask.task_id, 'task')" 
                                                    class="secondary-button">
                                                <i class="fas fa-share-alt mr-2"></i>
                                                {{ t('copyShareLink') }}
                                            </button>
                                            
                                            <button v-if="['SUCCEED', 'FAILED', 'CANCEL'].includes(modalTask?.status)"
                                                    @click="deleteTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="secondary-button delete-button">
                                                <i class="fas fa-trash mr-2"></i>
                                                {{ t('deleteTask') }}
                                            </button>
                                        </div>
                                        
                                        <!-- 技术信息 -->
                                        <div class="tech-info">
                                            <p class="tech-text">{{ t('poweredByLightX2V') }}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 详细信息面板 -->
                        <div class="details-panel">
                            <div class="details-content">
                                <!-- 任务状态显示 -->
                                <div class="bg-secondary/30 rounded-xl p-6 mb-6">
                                    <h4 class="text-sm font-medium mb-3 flex items-center">
                                        <i class="fas fa-info-circle text-gradient-primary mr-2"></i>
                                        {{ t('taskInfo') }}
                                    </h4>
                                    <ul class="space-y-2 text-sm">
                                        <li class="flex justify-between">
                                            <span class="text-gray-400">{{ t('taskID') }}</span>
                                            <span>{{ modalTask?.task_id }}</span>
                                        </li>
                                        <li class="flex justify-between">
                                            <span class="text-gray-400">{{ t('taskType') }}</span>
                                            <span>{{ getTaskTypeName(modalTask) }}</span>
                                        </li>
                                        <li class="flex justify-between">
                                            <span class="text-gray-400">{{ t('modelName') }}</span>
                                            <span class="text">{{ modalTask?.model_cls }}</span>
                                        </li>
                                        <li class="flex justify-between">
                                            <span class="text-gray-400">{{ t('createTime') }}</span>
                                            <span>{{ formatTime(modalTask?.create_t) }}</span>
                                        </li>
                                        <li class="flex justify-between">
                                            <span class="text-gray-400">{{ t('updateTime') }}</span>
                                            <span>{{ formatTime(modalTask?.update_t) }}</span>
                                        </li>
                                        <li class="flex justify-between">
                                            <span class="text-gray-400">{{ t('status') }}</span>
                                            <div class="flex items-center gap-2">
                                                <span :class="getStatusTextClass(modalTask?.status)">{{
                                                    getTaskStatusDisplay(modalTask?.status) }}</span>
                                                <button
                                                    v-if="modalTask?.status === 'FAILED' && getTaskFailureInfo(modalTask)"
                                                    @click="showFailureDetails = !showFailureDetails"
                                                    class="text-red-400 hover:text-red-300 transition-colors"
                                                    :title="t('viewFailureReason')">
                                                    <i class="fas fa-exclamation-triangle text-xs"></i>
                                                </button>
                                            </div>
                                        </li>
                                    </ul>

                                    <!-- 失败原因详情 -->
                                    <div v-if="showFailureDetails && modalTask?.status === 'FAILED' && getTaskFailureInfo(modalTask)"
                                         class="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                                        <div class="flex items-start gap-2">
                                            <i class="fas fa-exclamation-triangle text-red-400 text-sm mt-0.5"></i>
                                            <div class="flex-1">
                                                <p class="text-xs text-red-300 font-medium mb-1">{{ t('failureReason') }}:</p>
                                                <p class="text-xs text-red-200 whitespace-pre-wrap">{{
                                                    getTaskFailureInfo(modalTask) }}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>


                                <!-- 提示词 -->
                                <div class="bg-secondary/30 rounded-xl p-4 mb-6">
                                    <h4 class="text-sm font-medium mb-3 flex items-center">
                                        <i class="fas fa-file-alt text-gradient-primary mr-2"></i>
                                    {{ t('prompt') }}
                                    </h4>
                                    <div class="bg-dark-light rounded-lg p-4 text-sm text-gray-300">
                                    <p>{{ modalTask?.params?.prompt || t('noPrompt') }}</p>
                                    </div>
                                </div>

                            <!-- 上传素材 -->
                            <div v-if="modalTask?.inputs && Object.keys(modalTask.inputs).length"
                                class="bg-secondary/30 rounded-xl p-4 mb-6">
                                    <h4 class="text-sm font-medium mb-3 flex items-center">
                                        <i class="fas fa-upload text-gradient-primary mr-2"></i>
                                    {{ t('uploadMaterials') }}
                                    <span v-if="loadingTaskFiles" class="ml-2 text-xs text-gray-400">({{ t('loading') }}...)</span>
                                    </h4>
                                    <div class="space-y-3">
                                    <template v-for="(input, key) in modalTask.inputs" :key="key">
                                            <div class="flex items-center gap-3">
                                                <template v-if="key.includes('image')">
                                                    <i class="fas fa-image text-gradient-primary text-xl"></i>
                                                    <div class="flex items-center gap-2">
                                                    <span
                                                        v-if="!selectedTaskFiles.inputs[key] || !selectedTaskFiles.inputs[key].url"
                                                        class="text-gray-400 text-sm">{{ typeof input === 'string' ?
                                                        input : t('image') }}</span>
                                                        <div class="flex items-center gap-2 relative group">
                                                            <img v-if="selectedTaskFiles.inputs[key] && selectedTaskFiles.inputs[key].url"
                                                            :src="selectedTaskFiles.inputs[key].url" :alt="input"
                                                                class="w-20 h-20 object-cover rounded bg-dark-light border border-gray-700"
                                                                >
                                                            <div v-else-if="selectedTaskFiles.inputs[key] && selectedTaskFiles.inputs[key].error"
                                                                class="w-20 h-20 rounded bg-red-900/20 border border-red-500/30 flex items-center justify-center">
                                                                <i class="fas fa-exclamation-triangle text-red-400"></i>
                                                            </div>
                                                            <div v-else
                                                                class="w-20 h-20 rounded bg-gray-700 border border-gray-600 flex items-center justify-center">
                                                                <i class="fas fa-spinner fa-spin text-gray-400"></i>
                                                            </div>
                                                        <button
                                                            v-if="selectedTaskFiles.inputs[key] && selectedTaskFiles.inputs[key].url"
                                                                    @click="downloadFile(selectedTaskFiles.inputs[key])"
                                                                    class="text-xs px-2 py-1 rounded">
                                                            <i
                                                                class="fas fa-download mr-1 text-white opacity-30 hover:opacity-100 transition-opacity"></i>
                                                            </button>
                                                        </div>
                                                    </div>
                                                </template>
                                                <template v-else-if="key.includes('audio')">
                                                    <i class="fas fa-microphone text-gradient-primary text-xl"></i>
                                                    <div class="flex items-center gap-2">
                                                    <span
                                                        v-if="!selectedTaskFiles.inputs[key] || !selectedTaskFiles.inputs[key].url"
                                                        class="text-gray-400 text-sm">{{ typeof input === 'string' ?
                                                        input : t('audioFile') }}</span>
                                                        <div class="flex items-center gap-2">
                                                        <audio
                                                            v-if="selectedTaskFiles.inputs[key] && selectedTaskFiles.inputs[key].url"
                                                            :src="selectedTaskFiles.inputs[key].url" controls
                                                                class="h-8 text"
                                                                preload="metadata">
                                                            {{ t('browserNotSupported') }}
                                                            </audio>
                                                            <div v-else-if="selectedTaskFiles.inputs[key] && selectedTaskFiles.inputs[key].error"
                                                                class="h-8 px-3 rounded bg-red-900/20 border border-red-500/30 flex items-center">
                                                            <i
                                                                class="fas fa-exclamation-triangle text-red-400 text-xs"></i>
                                                            </div>
                                                            <div v-else
                                                                class="h-8 px-3 rounded bg-gray-700 border border-gray-600 flex items-center">
                                                                <i class="fas fa-spinner fa-spin text-gray-400 text-xs"></i>
                                                            </div>
                                                        <button
                                                            v-if="selectedTaskFiles.inputs[key] && selectedTaskFiles.inputs[key].url"
                                                                    @click="downloadFile(selectedTaskFiles.inputs[key])"
                                                                    class="text-xs px-2 py-1 rounded">
                                                            <i
                                                                class="fas fa-download mr-1 text-white opacity-30 hover:opacity-100 transition-opacity"></i>
                                                            </button>
                                                    </div>
                                </div>
                                                </template>
                                    </div>
                                        </template>
                                    </div>
                                </div>

                </div>
            </div>
        </div>
    </div>
</div>
</template>

<style scoped>
/* Landing Page 样式 */
.landing-page {
    min-height: calc(100vh - 60px);
    width: 100%;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: white;
    position: fixed;
    top: 60px;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 50;
}

.modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: white;
    margin: 0;
    display: flex;
    align-items: center;
}

.header-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

.action-button {
    background: none;
    border: none;
    color: #9ca3af;
    font-size: 1.25rem;
    cursor: pointer;
    transition: all 0.2s;
    padding: 0.5rem;
    border-radius: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.action-button:hover {
    color: white;
    background: rgba(255, 255, 255, 0.1);
}

.back-button:hover {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
}

.close-button:hover {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.main-content {
    width: 100%;
    padding: 2rem 0;
    min-height: calc(100vh - 80px);
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
}

/* 内容网格布局 */
.content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
    align-items: center;
    min-height: 60vh;
}


/* 视频区域 */
.video-section {
    display: flex;
    justify-content: center;
    align-items: center;
}


.video-container {
    width: 100%;
    max-width: 500px;
    aspect-ratio: 9/16;
    background: #000;
    border-radius: 1rem;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
    position: relative;
}

.video-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #1f2937;
}

.error-placeholder {
    background: #1f2937;
}

.error-icon {
    font-size: 2rem;
    color: #ef4444;
    margin-bottom: 1rem;
}

.cancel-placeholder {
    background: #1f2937;
}

.cancel-icon {
    font-size: 2rem;
    color: #f59e0b;
    margin-bottom: 1rem;
}

.loading-spinner {
    font-size: 2rem;
    color: #8b5cf6;
    margin-bottom: 1rem;
}

.loading-text {
    color: #9ca3af;
    font-size: 0.875rem;
}

.video-player {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* 信息区域 */
.info-section {
    display: flex;
    align-items: center;
    justify-content: center;
}

.info-content {
    max-width: 500px;
    padding: 2rem 0;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.main-description {
    font-size: 1.25rem;
    color: #d1d5db;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* 特性列表 */
.features-list {
    margin-bottom: 2.5rem;
}

.feature-item {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    padding: 0.75rem 0;
}

.feature-icon {
    width: 40px;
    height: 40px;
    background: rgba(139, 92, 246, 0.1);
    border-radius: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
    color: #8b5cf6;
    font-size: 1.125rem;
}

.feature-text {
    font-size: 1rem;
    color: #e5e7eb;
    font-weight: 500;
}

/* 操作按钮 */
.action-buttons {
    margin-bottom: 2rem;
}

.primary-button {
    width: 100%;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #8b5cf6, #a855f7);
    border: none;
    border-radius: 0.75rem;
    color: white;
    font-size: 1.125rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.primary-button:hover {
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.4);
}

.secondary-button {
    width: 100%;
    padding: 0.75rem 1.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.5rem;
    color: white;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.5rem;
}

.secondary-button:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
}

.delete-button {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
    color: #ef4444;
}

.delete-button:hover {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.5);
}

/* 技术信息 */
.tech-info {
    text-align: center;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.tech-text {
    color: #9ca3af;
    font-size: 0.875rem;
    font-weight: 500;
}

/* 详细信息面板 */
.details-panel {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    padding: 5rem 0;
}

.details-content {
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
}

/* 输入素材标题 */
.materials-header {
    text-align: center;
    margin-bottom: 2rem;
}

.materials-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
}

/* 三个并列的卡片 */
.materials-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
}

.material-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    overflow: hidden;
    transition: all 0.3s ease;
}

.material-card:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-2px);
}

.card-header {
    background: rgba(139, 92, 246, 0.1);
    padding: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    position: relative;
}

.card-icon {
    font-size: 1.25rem;
    color: #8b5cf6;
}

.card-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: white;
    margin: 0;
    flex: 1;
}

.card-actions {
    display: flex;
    gap: 0.5rem;
    margin-left: auto;
}

.action-btn {
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 0.375rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.875rem;
}

.download-btn {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.download-btn:hover {
    background: rgba(34, 197, 94, 0.3);
    border-color: rgba(34, 197, 94, 0.5);
    transform: scale(1.05);
}

.copy-btn {
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
    border: 1px solid rgba(107, 114, 128, 0.3);
}

.copy-btn:hover {
    background: rgba(107, 114, 128, 0.3);
    border-color: rgba(107, 114, 128, 0.5);
    transform: scale(1.05);
}

.card-content {
    padding: 1.5rem;
    min-height: 200px;
}

/* 图片网格 */
.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

.image-item {
    text-align: center;
}

.image-container {
    position: relative;
    width: 100%;
    min-height: 120px;
    margin-bottom: 0.5rem;
    border-radius: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: center;
}

.image-preview {
    max-width: 100%;
    min-height: 80px;
    height: auto;
    width: auto;
    object-fit: contain;
    display: block;
    position: relative !important;
}

/* 音频列表 */
.audio-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.audio-item {
    text-align: center;
}

.audio-player {
    width: 100%;
    height: 40px;
    margin-bottom: 0.5rem;
}

/* 提示词内容 */
.prompt-content {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.5rem;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.prompt-text {
    color: #d1d5db;
    line-height: 1.6;
    margin: 0;
    word-break: break-word;
}

/* 空状态 */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 120px;
    color: #6b7280;
}

.empty-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.5;
}

.empty-text {
    font-size: 0.875rem;
    margin: 0;
    opacity: 0.7;
}

/* 响应式设计 */
@media (max-width: 1024px) {
    .content-grid {
        gap: 3rem;
        padding: 0 1.5rem;
    }
    
    .main-title {
        font-size: 2.5rem;
    }
    
    .video-container {
        max-width: 400px;
    }
    
    /* 卡片响应式 */
    .materials-cards {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
}

@media (max-width: 768px) {
    .main-content {
        padding: 1rem 0;
    }
    
    .content-grid {
        grid-template-columns: 1fr;
        gap: 2rem;
        padding: 0 1rem;
    }
    
    .main-title {
        font-size: 2rem;
    }
    
    .main-description {
        font-size: 1.125rem;
    }
    
    .video-container {
        max-width: 300px;
    }
    
    .info-content {
        padding: 1rem 0;
    }
    
    .details-content {
        padding: 0 1rem;
    }
    
    /* 移动端卡片调整 */
    .materials-cards {
        gap: 1rem;
    }
    
    .card-content {
        padding: 1rem;
        min-height: 150px;
    }
    
    .image-grid {
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }
    
    .materials-title {
        font-size: 1.25rem;
    }
}
</style>