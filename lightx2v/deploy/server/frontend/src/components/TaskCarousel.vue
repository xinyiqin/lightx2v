<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import {
    getTaskFileUrlSync,
    getTaskFileUrl,
    getTaskTypeName,
    formatTime,
    getTaskStatusDisplay,
    getStatusTextClass,
    getProgressTitle,
    getProgressInfo,
    getOverallProgress,
    getSubtaskStatusText,
    getSubtaskProgress,
    formatEstimatedTime,
    showAlert,
    cancelTask,
    resumeTask,
    downloadFile,
    getTaskFileFromCache,
    apiRequest,
    copyShareLink,
    currentTask,
    startPollingTask
} from '../utils/other'

const { t } = useI18n()

// Props
const props = defineProps({
    tasks: {
        type: Array,
        required: true,
        default: () => []
    }
})

// 响应式数据å
const isVideoLoaded = ref(false)
const isVideoError = ref(false)
const videoElement = ref(null)

// 计算属性
const sortedTasks = computed(() => {
    // 按创建时间排序，最新的在前
    return [...props.tasks].sort((a, b) => {
        const timeA = new Date(a.created_at || a.task_id).getTime()
        const timeB = new Date(b.created_at || b.task_id).getTime()
        return timeB - timeA
    })
})

const taskStatus = computed(() => currentTask.value?.status || 'CREATED')
const isCompleted = computed(() => taskStatus.value === 'SUCCEED')
const isRunning = computed(() => ['CREATED', 'PENDING', 'RUNNING'].includes(taskStatus.value))
const isFailed = computed(() => taskStatus.value === 'FAILED')
const isCancelled = computed(() => taskStatus.value === 'CANCEL')

// 当前任务索引（用于显示）
const currentTaskIndex = computed(() => {
    return sortedTasks.value.findIndex(task => task.task_id === currentTask.value?.task_id)
})

// 获取视频URL
const videoUrl = computed(() => {
    if (!isCompleted.value || !currentTask.value) return null
    return getTaskFileUrlSync(currentTask.value.task_id, 'output_video')
})

// 获取图片URL（用于缩略图）
const imageUrl = computed(() => {
    if (!currentTask.value) return null
    return getTaskFileUrlSync(currentTask.value.task_id, 'input_image')
})

// 更新当前任务数据并启动轮询
const updateCurrentTaskData = async (task) => {
    if (!task?.task_id) return
    
    try {
        const response = await apiRequest(`/api/v1/task/query?task_id=${task.task_id}`)
        if (response && response.ok) {
            const updatedTask = await response.json()
            // 更新全局currentTask
            currentTask.value = updatedTask
            console.log('TaskCarousel: 更新任务数据', updatedTask)
            
            // 如果任务还在进行中，开始轮询状态
            if (['CREATED', 'PENDING', 'RUNNING'].includes(updatedTask.status)) {
                startPollingTask(updatedTask.task_id)
            }
        }
    } catch (error) {
        console.warn(`TaskCarousel: 获取任务数据失败 task_id=${task.task_id}`, error.message)
    }
}



// 任务切换方法
const goToPreviousTask = () => {
    if (sortedTasks.value.length <= 1) return
    
    const currentIndex = sortedTasks.value.findIndex(task => task.task_id === currentTask.value?.task_id)
    if (currentIndex === -1) return
    
    const newIndex = currentIndex > 0 ? currentIndex - 1 : sortedTasks.value.length - 1
    const newTask = sortedTasks.value[newIndex]
    currentTask.value = newTask
    resetVideoState()
    // 更新新任务的数据并启动轮询
    updateCurrentTaskData(newTask)
}

const goToNextTask = () => {
    if (sortedTasks.value.length <= 1) return
    
    const currentIndex = sortedTasks.value.findIndex(task => task.task_id === currentTask.value?.task_id)
    if (currentIndex === -1) return
    
    const newIndex = currentIndex < sortedTasks.value.length - 1 ? currentIndex + 1 : 0
    const newTask = sortedTasks.value[newIndex]
    currentTask.value = newTask
    resetVideoState()
    // 更新新任务的数据并启动轮询
    updateCurrentTaskData(newTask)
}

// 处理任务指示器点击
const handleTaskIndicatorClick = (task) => {
    currentTask.value = task
    resetVideoState()
    // 更新任务数据并启动轮询
    updateCurrentTaskData(task)
}

// 重置视频状态
const resetVideoState = () => {
    isVideoLoaded.value = false
    isVideoError.value = false
}

// 视频加载事件
const onVideoLoaded = () => {
    isVideoLoaded.value = true
    isVideoError.value = false
}

const onVideoError = () => {
    isVideoError.value = true
    isVideoLoaded.value = false
}

// 处理下载
const handleDownload = async () => {
    if (!currentTask.value?.outputs?.output_video) return
    
    try {
        await downloadFile(
            currentTask.value.task_id, 
            'output_video', 
            currentTask.value.outputs.output_video
        )
        showAlert('下载成功', 'success')
    } catch (error) {
        console.error('下载失败:', error)
        showAlert('下载失败，请重试', 'danger')
    }
}

// 处理取消任务
const handleCancel = async () => {
    if (!currentTask.value?.task_id) return
    
    try {
        await cancelTask(currentTask.value.task_id)
        showAlert('任务已取消', 'info')
    } catch (error) {
        console.error('取消任务失败:', error)
        showAlert('取消任务失败，请重试', 'danger')
    }
}

// 处理分享任务
const handleShareTask = async () => {
    if (!currentTask.value?.task_id) return
    
    try {
        await copyShareLink(currentTask.value.task_id, 'task')
        showAlert('分享链接已复制到剪贴板', 'success')
    } catch (error) {
        console.error('分享失败:', error)
        showAlert('分享失败，请重试', 'danger')
    }
}


// 获取文件扩展名
const getFileExtension = (fileKey) => {
    if (fileKey.includes('video')) return 'mp4'
    if (fileKey.includes('image')) return 'jpg'
    if (fileKey.includes('audio')) return 'mp3'
    return 'file'
}

// 处理文件下载
const handleDownloadFile = async (taskId, fileKey, fileName) => {
    try {
        console.log('开始下载文件:', { taskId, fileKey, fileName })

        // 处理文件名，确保有正确的后缀名
        let finalFileName = fileName
        if (fileName && typeof fileName === 'string') {
            // 检查是否已有后缀名
            const hasExtension = /\.[a-zA-Z0-9]+$/.test(fileName)
            if (!hasExtension) {
                // 没有后缀名，根据文件类型添加
                const extension = getFileExtension(fileKey)
                finalFileName = `${fileName}.${extension}`
                console.log('添加后缀名:', finalFileName)
            }
        } else {
            // 没有文件名，使用默认名称
            finalFileName = `${fileKey}.${getFileExtension(fileKey)}`
        }

        // 先尝试从缓存获取
        let fileData = getTaskFileFromCache(taskId, fileKey)
        console.log('缓存中的文件数据:', fileData)

        if (fileData && fileData.blob) {
            // 缓存中有blob数据，直接使用
            console.log('使用缓存中的文件数据')
            const url = URL.createObjectURL(fileData.blob)
            const a = document.createElement('a')
            a.href = url
            a.download = finalFileName
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
            showAlert('文件下载成功', 'success')
            return
        }

        // 缓存中没有，从服务器获取
        console.log('从服务器获取文件')
        const response = await apiRequest(`/api/v1/tasks/${taskId}/files/${fileKey}`, {
            method: 'GET',
            responseType: 'blob'
        })

        if (response && response.data) {
            // 创建下载链接
            const url = URL.createObjectURL(response.data)
            const a = document.createElement('a')
            a.href = url
            a.download = finalFileName
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
            showAlert('文件下载成功', 'success')
        } else {
            throw new Error('文件数据为空')
        }
    } catch (error) {
        console.error('下载文件失败:', error)
        showAlert(`下载失败: ${error.message}`, 'danger')
    }
}

// 键盘事件处理
const handleKeydown = (event) => {
    if (event.key === 'ArrowLeft') {
        goToPreviousTask()
    } else if (event.key === 'ArrowRight') {
        goToNextTask()
    }
}

// 生命周期
onMounted(() => {
    document.addEventListener('keydown', handleKeydown)
    // 初始化时设置第一个任务为当前任务
    if (sortedTasks.value.length > 0 && !currentTask.value) {
        const firstTask = sortedTasks.value[0]
        currentTask.value = firstTask
        // 更新任务数据并启动轮询
        updateCurrentTaskData(firstTask)
    }
})

onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
    <div class="task-carousel-container">
            <div class="task-counter">
                {{ currentTaskIndex + 1 }} / {{ sortedTasks.length }}
            </div>

        <div class="video-section">
            <!-- 导航箭头 -->
            <button 
                v-if="sortedTasks.length > 1"
                @click="goToPreviousTask"
                class="nav-button nav-button-left bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                :disabled="sortedTasks.length <= 1">
                <i class="fas fa-chevron-left"></i>
            </button>

            <button 
                v-if="sortedTasks.length > 1"
                @click="goToNextTask"
                class="nav-button nav-button-right bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                :disabled="sortedTasks.length <= 1">
                <i class="fas fa-chevron-right"></i>
            </button>

            <div class="video-container">
                <!-- 已完成：显示视频播放器 -->
                <video
                    v-if="isCompleted && videoUrl"
                    :src="videoUrl"
                    :poster="imageUrl"
                    class="video-player"
                    controls
                    loop
                    preload="metadata"
                    @loadeddata="onVideoLoaded"
                    @error="onVideoError"
                    ref="videoElement">
                    {{ t('browserNotSupported') }}
                </video>

                <!-- 进行中：显示图片缩略图 + 进度条 -->
                <div v-else-if="isRunning" class="video-placeholder">
                    <!-- 背景图片 -->
                    <div v-if="imageUrl" class="background-image">
                        <img :src="imageUrl" :alt="getTaskTypeName(currentTask?.task_type)" class="dimmed-image">
                    </div>
                    
                    <!-- 进度条 -->
                    <div class="progress-overlay">
                        <div class="progress-container">
                            
                            <!-- 进度条-->
                            <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(taskStatus)">
                                <div v-for="(subtask, index) in (currentTask?.subtasks || [])" :key="index">
                                    <!-- PENDING状态：显示排队信息 -->
                                    <div v-if="subtask.status === 'PENDING'" class="queue-info">
                                        <div v-if="subtask.estimated_pending_order !== null && subtask.estimated_pending_order !== undefined && subtask.estimated_pending_order >= 0" class="queue-visualization">
                                            <div class="queue-people">
                                                <i v-for="n in Math.min(Math.max(subtask.estimated_pending_order, 0), 10)"
                                                   :key="n"
                                                   class="fas fa-user queue-person"></i>
                                                <span v-if="subtask.estimated_pending_order > 10" class="queue-more">
                                                    +{{ subtask.estimated_pending_order - 10 }}
                                                </span>
                                            </div>
                                            <span class="queue-text">
                                                {{ t('queuePosition') }}: {{ subtask.estimated_pending_order }}
                                            </span>
                                        </div>
                                    </div>

                                    <!-- RUNNING状态：显示进度条 -->
                                    <div v-else-if="subtask.status === 'RUNNING'" class="progress-container">
                                        <div class="minimal-progress-bar">
                                            <div class="progress-line">
                                                <div class="progress-fill" :style="{ width: getSubtaskProgress(subtask) + '%' }"></div>
                                                <div class="moving-dot" :style="{ left: getSubtaskProgress(subtask) + '%' }"></div>
                                            </div>
                                        </div>
                                        <div class="progress-info">
                                            <span class="estimated-time">
                                                {{ getSubtaskProgress(subtask) }}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 失败：显示图片缩略图 + 错误信息 -->
                <div v-else-if="isFailed" class="video-placeholder error-placeholder">
                    <!-- 背景图片 -->
                    <div v-if="imageUrl" class="background-image">
                        <img :src="imageUrl" :alt="getTaskTypeName(currentTask?.task_type)" class="dimmed-image">
                    </div>
                    
                    <!-- 错误信息 -->
                    <div class="error-overlay">
                        <div class="error-icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <p class="error-text">{{ t('videoGeneratingFailed') }}</p>
                    </div>
                </div>

                <!-- 已取消：显示图片缩略图 + 取消信息 -->
                <div v-else-if="isCancelled" class="video-placeholder cancel-placeholder">
                    <!-- 背景图片 -->
                    <div v-if="imageUrl" class="background-image">
                        <img :src="imageUrl" :alt="getTaskTypeName(currentTask?.task_type)" class="dimmed-image">
                    </div>
                    
                    <!-- 取消信息 -->
                    <div class="cancel-overlay">
                        <div class="cancel-icon">
                            <i class="fas fa-ban"></i>
                        </div>
                        <p class="cancel-text">{{ t('taskCancelled') }}</p>
                    </div>
                </div>

                <!-- 默认状态 -->
                <div v-else class="video-placeholder">
                    <div class="loading-spinner">
                        <i class="fas fa-video"></i>
                    </div>
                    <p class="loading-text">{{ t('videoNotAvailable') }}</p>
                </div>
            </div>

            <!-- 操作按钮 -->
            <div class="action-buttons">
                <!-- 已完成：下载按钮 -->
                <button 
                    v-if="isCompleted && currentTask?.outputs?.output_video"
                    @click="handleDownloadFile(currentTask.task_id, 'output_video', currentTask.outputs.output_video)"
                    class="action-button bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                    :title="t('download')">
                    <i class="fas fa-download"></i>
                </button>

                <!-- 已完成：分享按钮 -->
                <button 
                    v-if="isCompleted && currentTask?.outputs?.output_video"
                    @click="handleShareTask"
                    class="action-button bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                    :title="t('share')">
                    <i class="fas fa-share-alt"></i>
                </button>

                <!-- 进行中：取消按钮 -->
                <button 
                    v-if="isRunning"
                    @click="handleCancel"
                    class="action-button bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                    :title="t('cancel')">
                    <i class="fas fa-times"></i>
                </button>

                <!-- 失败：重试按钮 -->
                <button 
                    v-if="isFailed"
                    @click="handleRetry"
                    class="action-button bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                    :title="t('retry')">
                    <i class="fas fa-redo"></i>
                </button>
            </div>
        </div>

        <!-- 任务指示器 -->
        <div v-if="sortedTasks.length > 1" class="task-indicators">
            <div 
                v-for="(task, index) in sortedTasks" 
                :key="task.task_id"
                @click="handleTaskIndicatorClick(task)"
                class="indicator hover:bg-laser-purple hover:scale-110"
                :class="index === currentTaskIndex? 'bg-laser-purple': 'bg-gray-400/30'">
            </div>
        </div>
    </div>
</template>

<style scoped>
.task-carousel-container {
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
}

.task-counter {
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 1rem;
}

.task-info {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.25rem;
}

.task-type {
    font-size: 0.75rem;
    color: #8b5cf6;
    font-weight: 500;
}

.task-time {
    font-size: 0.75rem;
    color: #9ca3af;
}

/* 视频区域 */
.video-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
    position: relative;
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

/* 导航按钮 */
.nav-button {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 50px;
    height: 50px;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    transition: all 0.2s ease;
    z-index: 10;
    backdrop-filter: blur(10px);
}


.nav-button:disabled {
    opacity: 0.3;
    cursor: not-allowed;
}

.nav-button-left {
    left: -60px;
}

.nav-button-right {
    right: -60px;
}

.video-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
}

.background-image {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
}

.dimmed-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.3) blur(2px);
}

.loading-spinner {
    font-size: 3rem;
    color: #8b5cf6;
    margin-bottom: 1rem;
    z-index: 2;
}

.loading-text {
    color: #9ca3af;
    font-size: 0.875rem;
    z-index: 2;
}

.video-player {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* 进度条覆盖层 */
.progress-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 2;
    padding: 2rem;
}

.progress-container {
    width: 100%;
    max-width: 300px;
    text-align: center;
}

.progress-status {
    font-size: 0.75rem;
    color: #8b5cf6;
    background: rgba(139, 92, 246, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 0.375rem;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

.progress-info {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    color: #ffffff;
}

.progress-text {
    flex: 1;
    text-align: left;
}

.progress-percentage {
    font-weight: 600;
    color: #8b5cf6;
}

.progress-bar {
    margin-top: 0.75rem;
}

.progress-line {
    position: relative;
    width: 100%;
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 1px;
}

.progress-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #8b5cf6, #a855f7);
    border-radius: 1px;
    transition: width 0.5s ease;
}

.moving-dot {
    position: absolute;
    top: -4px;
    width: 10px;
    height: 10px;
    background: linear-gradient(45deg, #8b5cf6, #a855f7);
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    transition: left 0.5s ease;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    }
    50% {
        transform: scale(1.2);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.8);
    }
}

/* TaskDetails进度条样式 */
.minimal-progress-bar {
    margin-bottom: 0.75rem;
}

.queue-info {
    margin-top: 0.75rem;
    text-align: center;
}

.queue-visualization {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.queue-people {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.25rem;
    margin-bottom: 0.5rem;
}

.queue-person {
    font-size: 0.875rem;
    color: #f59e0b;
    opacity: 0.8;
}

.queue-more {
    font-size: 0.75rem;
    color: #f59e0b;
    font-weight: 600;
    margin-left: 0.25rem;
}

.queue-text {
    font-size: 0.75rem;
    color: #f59e0b;
    font-weight: 500;
}

.estimated-time {
    display: flex;
    align-items: center;
    font-size: 0.875rem;
    color: #22c55e;
    font-weight: 600;
    animation: countdown 1s ease-in-out infinite;
}

@keyframes countdown {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.8;
        transform: scale(1.05);
    }
}

/* 错误状态 */
.error-placeholder {
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
}

.error-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 2;
    padding: 2rem;
}

.error-icon {
    font-size: 3rem;
    color: #ef4444;
    margin-bottom: 1rem;
    animation: pulse 2s infinite;
}

.error-text {
    color: #ffffff;
    font-size: 0.875rem;
    text-align: center;
    font-weight: 500;
}

/* 取消状态 */
.cancel-placeholder {
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
}

.cancel-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 2;
    padding: 2rem;
}

.cancel-icon {
    font-size: 3rem;
    color: #9ca3af;
    margin-bottom: 1rem;
}

.cancel-text {
    color: #ffffff;
    font-size: 0.875rem;
    text-align: center;
    font-weight: 500;
}

/* 操作按钮 */
.action-buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
}

.action-button {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    transition: all 0.2s ease;
    border: none;
    cursor: pointer;
    backdrop-filter: blur(10px);
}




/* 任务指示器 */
.task-indicators {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1rem;
}

.indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.2s ease;
}



/* 动画 */
@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.7;
    }
}

/* 响应式设计 */
@media (max-width: 768px) {    
    .video-container {
        max-width: 400px;
    }
    
    .nav-button-left {
        left: -30px;
    }
    
    .nav-button-right {
        right: -30px;
    }
    
    .progress-overlay,
    .error-overlay,
    .cancel-overlay {
        padding: 1rem;
    }
    
    .action-button {
        width: 45px;
        height: 45px;
        font-size: 1rem;
    }
}

@media (max-width: 480px) {
    .video-container {
        max-width: 300px;
    }
    .nav-button-left {
        left: -30px;
    }
    
    .nav-button-right {
        right: -30px;
    }
}
</style>
